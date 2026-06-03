#!/usr/bin/env python3
"""
Search Head with External Buffer — Post-Training (Experiment 3)

Loads a pretrained search head checkpoint and adds an external buffer search.
The backbone and original internal search are frozen. Only the expanded head
(which now accepts 3 embeddings) is trained.

Architecture:
  - Frozen backbone produces embeddings for both the training sequence and
    the external buffer (preceding context from the same document)
  - Internal search: same as pretrained (frozen), finds best_internal from h[0..t-1]
  - External search: finds best_external from the buffer embeddings
  - Expanded head: MLP([h[t], best_internal, best_external]) → logits

Data pipeline:
  - Fetch 3*N + 1 characters per sample (instead of N + 1)
  - First 2*N characters → run through frozen backbone → 2*N external buffer embeddings
  - Last N + 1 characters → training sequence (same as before)

The head's first linear layer is expanded from 2*D to 3*D columns:
  - First 2*D columns: initialized from pretrained weights (frozen during warmup)
  - Last D columns: initialized to zero (trained from scratch)
"""

import argparse
import math
import random
import time
from pathlib import Path

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

# ── Hyperparameters ──────────────────────────────────────────────────────
N = 256            # context window (block_size) for training sequence
BUFFER_SIZE = 256  # number of embeddings in external buffer
D = 512            # embedding dimension
N_HEAD = 8
N_LAYER = 8
W = 12             # local window size for recent heads
MLP_HEAD_HIDDEN = 2048
ALL_POS_WEIGHT = 1.0
LAST_POS_WEIGHT = 1.0
BATCH_SIZE = 10
LR = 1e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
EPOCHS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
CODE_DATASET_NAME = "codeparrot/codeparrot-clean"
CODE_TEXT_FIELD = "content"
VOCAB_SIZE = 256   # byte-level

BASE = Path(__file__).parent
CHECKPOINT_FILE = BASE / "checkpoints" / "best_model_ext_buffer.pt"
LATEST_CHECKPOINT_FILE = BASE / "checkpoints" / "latest_model_ext_buffer.pt"
PRETRAINED_CHECKPOINT = BASE / "checkpoints" / "best_model_byte.pt"

TOKENS_PER_EPOCH = 10_000_000
VAL_BATCHES = 200


# ── Data ─────────────────────────────────────────────────────────────────

def text_to_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class ExtBufferDataset(IterableDataset):
    """Yields (buffer_ids, seq_x, seq_y) tuples.

    Each sample fetches (block_size + buffer_size - 1) + seq_len + 1 consecutive characters:
      - First (block_size + buffer_size - 1) characters: raw buffer context
        (produces buffer_size embeddings via sliding window of block_size, stride 1,
         taking the last-position embedding from each window)
      - Last seq_len + 1 characters: training sequence (x = first N, y = last N)
    """

    def __init__(self, hf_dataset, seq_len: int = N, buffer_size: int = BUFFER_SIZE,
                 block_size: int = N, text_field: str = "text"):
        self.dataset = hf_dataset
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.text_field = text_field
        self.buf_chars = block_size + buffer_size - 1  # raw chars needed for buffer
        self.total_len = self.buf_chars + seq_len + 1

    def __iter__(self):
        for example in self.dataset:
            text = example.get(self.text_field) or example.get("content") or example.get("text") or ""
            ids = text_to_ids(text)
            stride = self.seq_len // 2
            for start in range(0, len(ids) - self.total_len + 1, stride):
                window = ids[start: start + self.total_len]
                if len(window) < self.total_len:
                    continue
                buf = torch.tensor(window[:self.buf_chars], dtype=torch.long)
                x = torch.tensor(window[self.buf_chars:self.buf_chars + self.seq_len], dtype=torch.long)
                y = torch.tensor(window[self.buf_chars + 1:self.buf_chars + self.seq_len + 1], dtype=torch.long)
                yield buf, x, y


def collate_ext_batch(batch):
    bufs, xs, ys = zip(*batch)
    return torch.stack(bufs), torch.stack(xs), torch.stack(ys)


# ── RMSNorm ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.weight


# ── Temporal Split Causal Multi-Head Attention ───────────────────────────

class TemporalSplitAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, local_window):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % 2 == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.local_window = local_window
        self.n_recent = n_head // 2
        self.n_historical = n_head - self.n_recent

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        recent = torch.zeros(block_size, block_size)
        for t in range(block_size):
            start = max(0, t - local_window + 1)
            recent[t, start:t + 1] = 1.0

        historical = torch.zeros(block_size, block_size)
        for t in range(block_size):
            end = t - local_window + 1
            if end > 0:
                historical[t, :end] = 1.0

        masks = torch.cat([
            recent.unsqueeze(0).expand(self.n_recent, -1, -1),
            historical.unsqueeze(0).expand(self.n_historical, -1, -1),
        ], dim=0)
        self.register_buffer("mask", masks.unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att.nan_to_num(0.0)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


# ── External Buffer Search Head GPT ─────────────────────────────────────

class ExternalBufferSearchGPT(nn.Module):
    """Search head with both internal and external buffer search.

    The backbone is shared (and frozen during post-training).
    The head takes 3 embeddings: [h[t], best_internal, best_external].
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, buffer_size,
                 mlp_head_hidden, local_window):
        super().__init__()
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': TemporalSplitAttention(n_embd, n_head, block_size, local_window),
                'ln1': RMSNorm(n_embd),
                'mlp': nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd, bias=False)
                ),
                'ln2': RMSNorm(n_embd)
            }) for _ in range(n_layer)
        ])

        self.ln_f = RMSNorm(n_embd)

        # Expanded head: takes 3 embeddings [h[t], best_internal, best_external] -> logits
        self.head = nn.Sequential(
            nn.Linear(3 * n_embd, mlp_head_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_head_hidden, vocab_size, bias=False),
        )

        # Internal search head (2*D -> hidden -> vocab) used for scoring internal candidates
        # This is loaded from the pretrained checkpoint and frozen
        self.internal_head = nn.Sequential(
            nn.Linear(2 * n_embd, mlp_head_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_head_hidden, vocab_size, bias=False),
        )

    def _get_embeddings(self, idx):
        """Run backbone on input ids. Works for any sequence length <= block_size."""
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            attn_out = block['attn'](block['ln1'](x))
            x = x + attn_out
            x = x + block['mlp'](block['ln2'](x))

        return self.ln_f(x)

    def _get_buffer_embeddings(self, buf_ids):
        """Produce buffer embeddings using a sliding window of block_size, stride 1.

        Each buffer embedding is the last-position output from a block_size window.
        buf_ids: (B, block_size + buffer_size - 1) raw token ids.
        Returns: (B, buffer_size, D) embeddings.
        """
        B, L = buf_ids.shape
        n_windows = L - self.block_size + 1  # == buffer_size

        # Build all overlapping windows: (B * n_windows, block_size)
        # Use unfold for efficiency
        windows = buf_ids.unfold(1, self.block_size, 1)  # (B, n_windows, block_size)
        windows_flat = windows.reshape(B * n_windows, self.block_size)

        # Process in sub-batches to avoid OOM
        sub_batch = max(1, 256 // self.block_size * 64)  # ~64 windows at a time
        sub_batch = min(sub_batch, 64)
        last_embs = []

        with torch.no_grad():
            for i in range(0, B * n_windows, sub_batch):
                chunk = windows_flat[i:i + sub_batch]
                embs = self._get_embeddings(chunk)  # (sub_batch, block_size, D)
                last_embs.append(embs[:, -1, :])    # take last position only

        all_last = torch.cat(last_embs, dim=0)  # (B * n_windows, D)
        return all_last.view(B, n_windows, -1)  # (B, buffer_size, D)

    def _internal_search(self, h):
        """Search over internal context (same as original search head). Returns best_j."""
        B, T, _ = h.shape
        rows, cols = torch.tril_indices(T, T, offset=-1, device=h.device)

        with torch.no_grad():
            last_embs = h[:, rows, :]
            prev_embs = h[:, cols, :]
            pairs = torch.cat([last_embs, prev_embs], dim=-1)

            logits = self.internal_head(pairs)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values

            score_matrix = torch.full((B, T, T), float('-inf'), device=h.device)
            score_matrix[:, rows, cols] = max_probs
            best_j = score_matrix[:, 1:, :].argmax(dim=-1)

        return best_j

    def _external_search(self, h_t, h_best_internal, h_buffer):
        """Search over external buffer using the expanded head.

        For each query position t, form triplet [h_t, h_best_internal, buf[j]]
        for every buffer position j, score through the head weights, and pick the
        buffer embedding that gives the highest confidence (max softmax prob).
        """
        B, T_minus_1, D = h_t.shape
        _, L, _ = h_buffer.shape

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            # Disable autocast here to prevent weight-cast caching that would
            # poison the subsequent grad-enabled call to self.head().
            h_t_f = h_t.float()
            h_best_internal_f = h_best_internal.float()
            h_buffer_f = h_buffer.float()

            # Expand for all buffer positions: (B, T-1, L, D)
            t_expanded = h_t_f.unsqueeze(2).expand(B, T_minus_1, L, D)
            int_expanded = h_best_internal_f.unsqueeze(2).expand(B, T_minus_1, L, D)
            buf_expanded = h_buffer_f.unsqueeze(1).expand(B, T_minus_1, L, D)
            triplets = torch.cat([t_expanded, int_expanded, buf_expanded], dim=-1)  # (B, T-1, L, 3D)

            # Score using head weights directly
            triplets_flat = triplets.view(B * T_minus_1 * L, 3 * D)
            hidden = F.gelu(F.linear(triplets_flat, self.head[0].weight.float()))
            logits = F.linear(hidden, self.head[2].weight.float())
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1).values   # (B * (T-1) * L)
            max_probs = max_probs.view(B, T_minus_1, L)

            best_ext = max_probs.argmax(dim=-1)  # (B, T-1)

        return best_ext

    def forward(self, seq_ids, buf_ids, targets=None):
        """
        Args:
            seq_ids: (B, N) training sequence token ids
            buf_ids: (B, BUFFER_SIZE) external buffer token ids
            targets: (B, N) next-token targets
        """
        B, T = seq_ids.shape

        # Get embeddings for training sequence (backbone is frozen, no grad needed)
        with torch.no_grad():
            h = self._get_embeddings(seq_ids)

        # Get embeddings for external buffer (frozen backbone, chunked)
        h_buffer = self._get_buffer_embeddings(buf_ids)

        # Internal search (frozen)
        best_j_internal = self._internal_search(h)

        # Gather internal search results first (needed for external search)
        query_pos = torch.arange(1, T, device=seq_ids.device).unsqueeze(0).expand(B, -1)
        batch_idx = torch.arange(B, device=seq_ids.device).unsqueeze(1).expand(-1, T - 1)

        h_t = h[batch_idx, query_pos, :]
        h_best_internal = h[batch_idx, best_j_internal, :]

        # External search using expanded head: score [h_t, h_best_internal, buf[j]]
        best_j_external = self._external_search(h_t, h_best_internal, h_buffer)
        h_best_external = h_buffer[batch_idx, best_j_external, :]

        # Concatenate all three and pass through expanded head
        triplet = torch.cat([h_t, h_best_internal, h_best_external], dim=-1)  # (B, T-1, 3D)
        logits = self.head(triplet)

        # Internal-only logits (what the pretrained model would predict without external buffer)
        with torch.no_grad():
            internal_pair = torch.cat([h_t, h_best_internal], dim=-1)  # (B, T-1, 2D)
            logits_internal = self.internal_head(internal_pair)

        loss = None
        loss_all = None
        loss_last = None
        loss_internal = None
        loss_last_internal = None
        if targets is not None:
            valid_targets = targets[:, 1:]
            loss_all = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                valid_targets.reshape(-1),
            )
            loss_last = F.cross_entropy(logits[:, -1, :], valid_targets[:, -1])
            loss = ALL_POS_WEIGHT * loss_all + LAST_POS_WEIGHT * loss_last
            with torch.no_grad():
                loss_internal = F.cross_entropy(
                    logits_internal.reshape(-1, logits_internal.size(-1)),
                    valid_targets.reshape(-1),
                )
                loss_last_internal = F.cross_entropy(
                    logits_internal[:, -1, :], valid_targets[:, -1]
                )

        return logits, loss, loss_all, loss_last, best_j_internal, best_j_external, logits_internal, loss_internal, loss_last_internal


# ── Load pretrained and expand ──────────────────────────────────────────

def load_pretrained_and_expand(pretrained_path, device, buffer_size=BUFFER_SIZE):
    """Load pretrained SearchHeadGPT and create ExternalBufferSearchGPT with expanded head."""

    # Load pretrained state dict
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_state = ckpt["model"]

    # Create new model
    model = ExternalBufferSearchGPT(
        vocab_size=VOCAB_SIZE, n_embd=D, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=N, buffer_size=buffer_size, mlp_head_hidden=MLP_HEAD_HIDDEN,
        local_window=W,
    ).to(device)

    # Load backbone weights (token_embedding, pos_embedding, blocks, ln_f)
    new_state = model.state_dict()
    for key in pretrained_state:
        if key.startswith("head."):
            continue  # handle head separately
        if key in new_state:
            new_state[key] = pretrained_state[key]

    # Load internal_head from pretrained head (frozen copy for scoring)
    for key in pretrained_state:
        if key.startswith("head."):
            internal_key = key.replace("head.", "internal_head.")
            if internal_key in new_state:
                new_state[internal_key] = pretrained_state[key]

    # Expand head.0.weight from (hidden, 2*D) to (hidden, 3*D)
    # First 2*D columns: pretrained weights.
    # Last D columns: 0.1 * the "best match" columns (D:2D) — warm start for external search.
    pretrained_w0 = pretrained_state["head.0.weight"]  # (MLP_HEAD_HIDDEN, 2*D)
    new_w0 = torch.zeros(MLP_HEAD_HIDDEN, 3 * D, device=device)
    new_w0[:, :2 * D] = pretrained_w0
    new_w0[:, 2 * D:] = 0.1 * pretrained_w0[:, D:2 * D]
    new_state["head.0.weight"] = new_w0

    # head.2.weight (hidden -> vocab): copy directly
    new_state["head.2.weight"] = pretrained_state["head.2.weight"]

    model.load_state_dict(new_state)

    # Freeze everything except the new D columns of head.0.weight
    for param in model.parameters():
        param.requires_grad = False
    model.head[0].weight.requires_grad = True

    # Gradient hook: zero out gradients for pretrained columns (first 2*D)
    def _mask_pretrained_cols(grad):
        grad[:, :2 * D] = 0.0
        return grad
    model.head[0].weight.register_hook(_mask_pretrained_cols)

    n_trainable = model.head[0].weight[:, 2*D:].numel()  # effective trainable
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Loaded pretrained from: {pretrained_path}")
    print(f"Total params: {n_total:,} | Trainable (new ext columns only): {n_trainable:,}")
    print(f"Frozen: backbone, internal_head, head.2, head.0[:,:2D] | Trainable: head.0[:,2D:] ({MLP_HEAD_HIDDEN}×{D})")

    return model


# ── Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_bytes: list[int], buffer_bytes: list[int],
             max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 40):
    model.eval()
    generated = list(prompt_bytes)
    block_size = model.block_size

    # Compute buffer embeddings once
    # Need block_size + buffer_size - 1 raw chars for the sliding window
    buf_raw_len = model.block_size + model.buffer_size - 1
    buf_t = torch.tensor([buffer_bytes[-buf_raw_len:]], dtype=torch.long, device=DEVICE)
    h_buffer = model._get_buffer_embeddings(buf_t)

    for _ in range(max_new_tokens):
        ctx = generated[-block_size:]
        if len(ctx) < 2:
            break
        x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        B, T = x.shape
        h = model._get_embeddings(x)

        t = T - 1
        last_emb = h[:, t:t+1, :]  # (1, 1, D)

        # Internal search: find best from h[0..t-1]
        prev_embs = h[:, :t, :]  # (1, t, D)
        internal_pairs = torch.cat([last_emb.expand(1, t, -1), prev_embs], dim=-1)
        int_logits = model.internal_head(internal_pairs)
        int_probs = F.softmax(int_logits, dim=-1)
        int_max_probs = int_probs.max(dim=-1).values
        best_int = int_max_probs.argmax(dim=-1)  # (1,)
        h_best_int = prev_embs[0, best_int[0], :].unsqueeze(0).unsqueeze(0)

        # External search: find best from buffer using expanded head
        L = h_buffer.size(1)
        triplets_ext = torch.cat([
            last_emb.expand(1, L, -1),
            h_best_int.expand(1, L, -1),
            h_buffer,
        ], dim=-1)
        ext_logits = model.head(triplets_ext)
        ext_probs = F.softmax(ext_logits, dim=-1)
        ext_max_probs = ext_probs.max(dim=-1).values
        best_ext = ext_max_probs.argmax(dim=-1)
        h_best_ext = h_buffer[0, best_ext[0], :].unsqueeze(0).unsqueeze(0)

        # Expanded head
        triplet = torch.cat([last_emb, h_best_int, h_best_ext], dim=-1)
        logits = model.head(triplet)[0, 0, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_byte = torch.multinomial(probs, 1).item()
        generated.append(next_byte)

        if len(generated) >= 2 and generated[-1] == 10 and generated[-2] == 10:
            break

    return bytes(generated)


# ── Training utilities ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_batches, use_amp):
    model.eval()
    total_loss = 0.0
    total_loss_last = 0.0
    total_loss_internal = 0.0
    total_loss_last_internal = 0.0
    total_tokens = 0
    total_batches = 0
    total_correct = 0
    total_correct_last = 0
    total_correct_internal = 0
    total_correct_last_internal = 0

    for buf, xb, yb in val_batches:
        buf, xb, yb = buf.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
        with torch.amp.autocast(DEVICE, enabled=use_amp):
            logits, _, loss_all, loss_last, _, _, logits_internal, loss_internal, loss_last_internal = model(xb, buf, targets=yb)

        valid_targets = yb[:, 1:]
        batch_tokens = valid_targets.numel()
        total_loss += loss_all.item() * batch_tokens
        total_loss_last += loss_last.item() * xb.size(0)
        total_loss_internal += loss_internal.item() * batch_tokens
        total_loss_last_internal += loss_last_internal.item() * xb.size(0)
        total_tokens += batch_tokens
        total_batches += xb.size(0)
        total_correct += (logits.argmax(dim=-1) == valid_targets).sum().item()
        total_correct_last += (logits[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()
        total_correct_internal += (logits_internal.argmax(dim=-1) == valid_targets).sum().item()
        total_correct_last_internal += (logits_internal[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_last = total_loss_last / max(total_batches, 1)
    avg_loss_internal = total_loss_internal / max(total_tokens, 1)
    avg_last_internal = total_loss_last_internal / max(total_batches, 1)
    acc = 100.0 * total_correct / max(total_tokens, 1)
    acc_last = 100.0 * total_correct_last / max(total_batches, 1)
    acc_internal = 100.0 * total_correct_internal / max(total_tokens, 1)
    acc_last_internal = 100.0 * total_correct_last_internal / max(total_batches, 1)
    return avg_loss, avg_last, acc, acc_last, avg_loss_internal, avg_last_internal, acc_internal, acc_last_internal


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-train Search Head with External Buffer")
    parser.add_argument("--pretrained", type=str, default=str(PRETRAINED_CHECKPOINT),
                        help="Path to pretrained search head checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--tokens-per-epoch", type=int, default=TOKENS_PER_EPOCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE,
                        help="External buffer size in characters")
    parser.add_argument("--wandb-project", type=str, default="search-head")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--generate-every", type=int, default=1, help="Generate sample every N epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    global DEVICE
    if args.cpu:
        DEVICE = "cpu"

    buffer_size = args.buffer_size
    batch_size = args.batch_size
    epochs = args.epochs
    tokens_per_epoch = args.tokens_per_epoch

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load pretrained model and expand head
    if args.resume and LATEST_CHECKPOINT_FILE.is_file():
        print("Resuming from latest external buffer checkpoint...")
        ckpt = torch.load(LATEST_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        model = ExternalBufferSearchGPT(
            vocab_size=VOCAB_SIZE, n_embd=D, n_head=N_HEAD, n_layer=N_LAYER,
            block_size=N, buffer_size=buffer_size, mlp_head_hidden=MLP_HEAD_HIDDEN,
            local_window=W,
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        # Freeze everything except new columns of head.0.weight
        for param in model.parameters():
            param.requires_grad = False
        model.head[0].weight.requires_grad = True

        def _mask_pretrained_cols(grad):
            grad[:, :2 * D] = 0.0
            return grad
        model.head[0].weight.register_hook(_mask_pretrained_cols)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")
    else:
        model = load_pretrained_and_expand(args.pretrained, DEVICE, buffer_size)
        start_epoch = 1
        best_val = float("inf")

    run_name = args.wandb_name or f"extBuffer-{buffer_size}-N{N}-D{D}-L{N_LAYER}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "version": "ext-buffer-v1",
            "experiment": 3,
            "N": N, "D": D, "N_HEAD": N_HEAD, "N_LAYER": N_LAYER,
            "BUFFER_SIZE": buffer_size, "W": W, "MLP_HEAD_HIDDEN": MLP_HEAD_HIDDEN,
            "HEAD_INPUT": f"3*D = {3*D}",
            "ALL_POS_WEIGHT": ALL_POS_WEIGHT,
            "LAST_POS_WEIGHT": LAST_POS_WEIGHT,
            "LR": LR, "WARMUP_STEPS": WARMUP_STEPS, "WEIGHT_DECAY": WEIGHT_DECAY,
            "BATCH_SIZE": batch_size, "EPOCHS": epochs,
            "TOKENS_PER_EPOCH": tokens_per_epoch,
            "VOCAB_SIZE": VOCAB_SIZE,
            "PRETRAINED": args.pretrained,
        },
        resume="allow" if args.resume else None,
    )

    print("Loading FineWeb-Edu (streaming)...")
    hf_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    print("Loading CodeParrot (streaming)...")
    code_dataset = load_dataset(CODE_DATASET_NAME, split="train", streaming=True, revision="refs/convert/parquet")

    buf_chars = N + buffer_size - 1
    print(f"Buffer size: {buffer_size} embeddings | Block size: {N} | Buffer raw chars: {buf_chars}")
    print(f"Data: {buf_chars + N + 1} chars per sample ({buf_chars} buffer + {N}+1 training)")
    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # Only optimize head parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTokens per epoch: {tokens_per_epoch:,}  |  Epochs: {epochs}\n")

    global_step = 0
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}")
        epoch_seed = random.randint(0, 2**31)
        shuffled_ds = hf_dataset.shuffle(seed=epoch_seed, buffer_size=10_000)
        shuffled_code = code_dataset.shuffle(seed=epoch_seed, buffer_size=10_000)

        loader_kwargs = dict(
            batch_size=batch_size,
            collate_fn=collate_ext_batch,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        fineweb_loader = iter(DataLoader(
            ExtBufferDataset(shuffled_ds, seq_len=N, buffer_size=buffer_size, block_size=N, text_field="text"),
            **loader_kwargs,
        ))
        code_loader = iter(DataLoader(
            ExtBufferDataset(shuffled_code, seq_len=N, buffer_size=buffer_size, block_size=N, text_field=CODE_TEXT_FIELD),
            **loader_kwargs,
        ))

        model.train()

        total_loss_all = 0.0
        total_loss_last = 0.0
        total_loss_internal = 0.0
        total_loss_last_internal = 0.0
        total_tokens = 0
        total_batches_count = 0
        total_correct = 0
        total_correct_last = 0
        total_correct_internal = 0
        total_correct_last_internal = 0
        epoch_bytes = 0
        val_batches = []
        tic_epoch = time.time()
        step = 0

        while True:
            step += 1
            try:
                if step % 2 == 1:
                    buf, xb, yb = next(fineweb_loader)
                else:
                    buf, xb, yb = next(code_loader)
            except StopIteration:
                break

            if step <= VAL_BATCHES:
                val_batches.append((buf.clone(), xb.clone(), yb.clone()))
                continue

            global_step += 1
            if global_step <= WARMUP_STEPS:
                warmup_lr = LR * global_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            buf, xb, yb = buf.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)

            with torch.amp.autocast(DEVICE, enabled=use_amp):
                logits, loss, loss_all, loss_last, _, _, logits_internal, loss_internal, loss_last_internal = model(xb, buf, targets=yb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            valid_targets = yb[:, 1:]
            batch_tokens = valid_targets.numel()
            total_loss_all += loss_all.item() * batch_tokens
            total_loss_last += loss_last.item() * xb.size(0)
            total_tokens += batch_tokens
            total_batches_count += xb.size(0)
            total_correct += (logits.argmax(dim=-1) == valid_targets).sum().item()

            with torch.no_grad():
                total_correct_last += (logits[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()
                total_loss_internal += loss_internal.item() * batch_tokens
                total_loss_last_internal += loss_last_internal.item() * xb.size(0)
                total_correct_internal += (logits_internal.argmax(dim=-1) == valid_targets).sum().item()
                total_correct_last_internal += (logits_internal[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()

            epoch_bytes += xb.numel()

            if step % 50 == 0 or step <= 3:
                avg_all = total_loss_all / max(total_tokens, 1)
                avg_last = total_loss_last / max(total_batches_count, 1)
                avg_int = total_loss_internal / max(total_tokens, 1)
                avg_int_last = total_loss_last_internal / max(total_batches_count, 1)
                acc = 100.0 * total_correct / max(total_tokens, 1)
                acc_last = 100.0 * total_correct_last / max(total_batches_count, 1)
                acc_int = 100.0 * total_correct_internal / max(total_tokens, 1)
                acc_int_last = 100.0 * total_correct_last_internal / max(total_batches_count, 1)
                pct = epoch_bytes / tokens_per_epoch * 100
                print(
                    f"  step {step:5d}  "
                    f"EXT all={avg_all:.4f} last={avg_last:.4f} acc={acc:.1f}%/{acc_last:.1f}% bpc={avg_all/math.log(2):.3f}/{avg_last/math.log(2):.3f}  "
                    f"INT all={avg_int:.4f} last={avg_int_last:.4f} acc={acc_int:.1f}%/{acc_int_last:.1f}% bpc={avg_int/math.log(2):.3f}/{avg_int_last/math.log(2):.3f}  "
                    f"{pct:.0f}%",
                    end="\r", flush=True,
                )

            if epoch_bytes >= tokens_per_epoch:
                break

        scheduler.step()

        train_loss = total_loss_all / max(total_tokens, 1)
        train_loss_last = total_loss_last / max(total_batches_count, 1)
        train_loss_internal = total_loss_internal / max(total_tokens, 1)
        train_loss_last_internal = total_loss_last_internal / max(total_batches_count, 1)
        train_acc = 100.0 * total_correct / max(total_tokens, 1)
        train_acc_last = 100.0 * total_correct_last / max(total_batches_count, 1)
        train_acc_internal = 100.0 * total_correct_internal / max(total_tokens, 1)
        train_acc_last_internal = 100.0 * total_correct_last_internal / max(total_batches_count, 1)

        val_loss, val_last, val_acc, val_acc_last, val_loss_internal, val_last_internal, val_acc_internal, val_acc_last_internal = evaluate(model, val_batches, use_amp)

        toc_epoch = time.time()
        print(
            f"\nEpoch {epoch:3d}  time={toc_epoch - tic_epoch:.0f}s"
            f"\n  AFTER ext search  | train: loss={train_loss:.4f}/{train_loss_last:.4f}  acc={train_acc:.1f}%/{train_acc_last:.1f}%  bpc={train_loss/math.log(2):.3f}/{train_loss_last/math.log(2):.3f}"
            f"\n                    | val:   loss={val_loss:.4f}/{val_last:.4f}  acc={val_acc:.1f}%/{val_acc_last:.1f}%  bpc={val_loss/math.log(2):.3f}/{val_last/math.log(2):.3f}"
            f"\n  BEFORE ext search | train: loss={train_loss_internal:.4f}/{train_loss_last_internal:.4f}  acc={train_acc_internal:.1f}%/{train_acc_last_internal:.1f}%  bpc={train_loss_internal/math.log(2):.3f}/{train_loss_last_internal/math.log(2):.3f}"
            f"\n                    | val:   loss={val_loss_internal:.4f}/{val_last_internal:.4f}  acc={val_acc_internal:.1f}%/{val_acc_last_internal:.1f}%  bpc={val_loss_internal/math.log(2):.3f}/{val_last_internal/math.log(2):.3f}",
            flush=True,
        )

        if epoch % args.generate_every == 0:
            prompt = b"The meaning of life is"
            buffer_ctx = b"Philosophy is the study of general and fundamental questions about existence, knowledge, values, reason, mind, and language. Such questions are often posed as problems to be studied or resolved. Some sources claim the term was coined by Pythagoras. " * 3
            sample = generate(model, list(prompt), list(buffer_ctx), max_new_tokens=150)
            sample_text = sample.decode("utf-8", errors="replace")
            print(f"  Sample: {sample_text[-200:]}")
            wandb.log({"sample": wandb.Html(f"<pre>{sample_text[-500:]}</pre>")}, commit=False)

        wandb.log({
            "epoch": epoch,
            "train/loss_all": train_loss,
            "train/loss_last": train_loss_last,
            "train/loss_internal": train_loss_internal,
            "train/loss_last_internal": train_loss_last_internal,
            "train/bpc": train_loss / math.log(2),
            "train/bpc_last": train_loss_last / math.log(2),
            "train/bpc_internal": train_loss_internal / math.log(2),
            "train/bpc_last_internal": train_loss_last_internal / math.log(2),
            "train/acc": train_acc,
            "train/acc_last": train_acc_last,
            "train/acc_internal": train_acc_internal,
            "train/acc_last_internal": train_acc_last_internal,
            "val/loss_all": val_loss,
            "val/loss_last": val_last,
            "val/loss_internal": val_loss_internal,
            "val/loss_last_internal": val_last_internal,
            "val/bpc": val_loss / math.log(2),
            "val/bpc_last": val_last / math.log(2),
            "val/bpc_internal": val_loss_internal / math.log(2),
            "val/bpc_last_internal": val_last_internal / math.log(2),
            "val/acc": val_acc,
            "val/acc_last": val_acc_last,
            "val/acc_internal": val_acc_internal,
            "val/acc_last_internal": val_acc_last_internal,
            "lr": scheduler.get_last_lr()[0],
        })

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "best_val": best_val,
        }, LATEST_CHECKPOINT_FILE)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_val": best_val,
            }, CHECKPOINT_FILE)
            print(f"  -> Saved best (val_bpc={val_loss / math.log(2):.3f})")

    wandb.finish()


if __name__ == "__main__":
    main()
