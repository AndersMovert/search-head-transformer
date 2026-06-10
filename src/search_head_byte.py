#!/usr/bin/env python3
"""
Search Head Transformer — Byte-Level (V=256)

Decoder transformer with confidence-based search at the output layer.
Instead of a linear projection h[t] → logits, the search head:
  1. Pairs h[t] with every previous embedding h[j] (j < t)
  2. Scores each pair by max softmax probability (confidence)
  3. Selects the highest-confidence pair for prediction

Uses Temporal Split Attention: half the heads attend to the last W positions
(local syntax), half attend only to older positions (long-range dependencies).

Trains on FineWeb-Edu (odd steps) and CodeParrot (even steps).
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
N = 256            # context window (block_size)
D = 512            # embedding dimension
N_HEAD = 8
N_LAYER = 8
W = 12             # local window size for recent heads
MLP_HEAD_HIDDEN = 2048
LOSS_MASK_POSITIONS = 64  # skip loss for the first N positions
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
CHECKPOINT_FILE = BASE / "checkpoints" / "best_model_byte.pt"
LATEST_CHECKPOINT_FILE = BASE / "checkpoints" / "latest_model_byte.pt"

TOKENS_PER_EPOCH = 10_000_000
VAL_BATCHES = 200


# ── Data ─────────────────────────────────────────────────────────────────

def text_to_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class TextDataset(IterableDataset):
    def __init__(self, hf_dataset, seq_len: int = N, text_field: str = "text"):
        self.dataset = hf_dataset
        self.seq_len = seq_len
        self.text_field = text_field

    def __iter__(self):
        for example in self.dataset:
            text = example.get(self.text_field) or example.get("content") or example.get("text") or ""
            ids = text_to_ids(text)
            total_len = self.seq_len + 1
            stride = self.seq_len // 2
            for start in range(0, len(ids) - total_len + 1, stride):
                window = ids[start: start + total_len]
                if len(window) < total_len:
                    continue
                x = torch.tensor(window[:self.seq_len], dtype=torch.long)
                y = torch.tensor(window[1:self.seq_len + 1], dtype=torch.long)
                yield x, y


def collate_batch(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


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
    """Multi-head attention with hard temporal masks.

    First half of heads ("recent"): attend only to the last W positions.
    Second half ("historical"): attend only to positions older than W.
    Both groups remain causal.
    """

    def __init__(self, n_embd, n_head, block_size, local_window):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % 2 == 0, "n_head must be even for 50/50 split"
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


# ── Search Head GPT ─────────────────────────────────────────────────────

class SearchHeadGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, mlp_head_hidden, local_window):
        super().__init__()
        self.block_size = block_size
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

        # Search head: takes 2 embeddings (query + best match) -> logits
        self.head = nn.Sequential(
            nn.Linear(2 * n_embd, mlp_head_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_head_hidden, vocab_size, bias=False),
        )

    def _get_embeddings(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            attn_out = block['attn'](block['ln1'](x))
            x = x + attn_out
            x = x + block['mlp'](block['ln2'](x))

        return self.ln_f(x)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        h = self._get_embeddings(idx)

        rows, cols = torch.tril_indices(T, T, offset=-1, device=idx.device)

        # Phase 1: Search — find best candidate for each query position (no grad)
        with torch.no_grad():
            last_embs_all = h[:, rows, :]
            prev_embs_all = h[:, cols, :]
            pairs_all = torch.cat([last_embs_all, prev_embs_all], dim=-1)

            all_logits_ng = self.head(pairs_all)
            probs = F.softmax(all_logits_ng, dim=-1)
            max_probs = probs.max(dim=-1).values

            score_matrix = torch.full((B, T, T), float('-inf'), device=idx.device)
            score_matrix[:, rows, cols] = max_probs

            best_j = score_matrix[:, 1:, :].argmax(dim=-1)

        # Phase 2: Compute logits for selected pairs only (with grad)
        query_pos = torch.arange(1, T, device=idx.device).unsqueeze(0).expand(B, -1)
        batch_idx = torch.arange(B, device=idx.device).unsqueeze(1).expand(-1, T - 1)

        last_embs_sel = h[batch_idx, query_pos, :]
        prev_embs_sel = h[batch_idx, best_j, :]
        selected_pairs = torch.cat([last_embs_sel, prev_embs_sel], dim=-1)

        logits = self.head(selected_pairs)

        loss = None
        loss_all = None
        loss_last = None
        if targets is not None:
            valid_targets = targets[:, 1:]
            # Mask first LOSS_MASK_POSITIONS from loss
            mask_start = max(0, LOSS_MASK_POSITIONS - 1)
            masked_logits = logits[:, mask_start:, :]
            masked_targets = valid_targets[:, mask_start:]
            loss_all = F.cross_entropy(
                masked_logits.reshape(-1, masked_logits.size(-1)),
                masked_targets.reshape(-1),
            )
            loss_last = F.cross_entropy(logits[:, -1, :], valid_targets[:, -1])
            loss = ALL_POS_WEIGHT * loss_all + LAST_POS_WEIGHT * loss_last

        return logits, loss, loss_all, loss_last, best_j


# ── Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_bytes: list[int], max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 40):
    model.eval()
    generated = list(prompt_bytes)
    block_size = model.block_size

    for _ in range(max_new_tokens):
        ctx = generated[-block_size:]
        if len(ctx) < 2:
            break
        x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        B, T = x.shape
        h = model._get_embeddings(x)

        t = T - 1
        last_emb = h[:, t:t+1, :].expand(1, t, -1)
        prev_embs = h[:, :t, :]
        pairs = torch.cat([last_emb, prev_embs], dim=-1)

        logits_all = model.head(pairs)
        probs_all = F.softmax(logits_all, dim=-1)
        max_probs = probs_all.max(dim=-1).values
        best = max_probs.argmax(dim=-1)

        logits = logits_all[0, best[0], :] / temperature

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
    total_tokens = 0
    total_batches = 0
    total_correct = 0
    total_correct_last = 0

    for xb, yb in val_batches:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.amp.autocast(DEVICE, enabled=use_amp):
            logits, _, loss_all, loss_last, _ = model(xb, targets=yb)

        valid_targets = yb[:, 1:]
        mask_start = max(0, LOSS_MASK_POSITIONS - 1)
        masked_targets = valid_targets[:, mask_start:]
        batch_tokens = masked_targets.numel()
        total_loss += loss_all.item() * batch_tokens
        total_loss_last += loss_last.item() * xb.size(0)
        total_tokens += batch_tokens
        total_batches += xb.size(0)
        masked_logits = logits[:, mask_start:, :]
        total_correct += (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
        total_correct_last += (logits[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    avg_last = total_loss_last / max(total_batches, 1)
    acc = 100.0 * total_correct / max(total_tokens, 1)
    acc_last = 100.0 * total_correct_last / max(total_batches, 1)
    return avg_loss, avg_last, acc, acc_last


def rebuild_optimizer(model, lr, weight_decay):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=lr)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Search-Head GPT (Byte-Level)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--load-weights", type=str, default=None,
                        help="Load model weights from checkpoint (starts training from epoch 1)")
    parser.add_argument("--tokens-per-epoch", type=int, default=TOKENS_PER_EPOCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--local-window", type=int, default=W, help="Local window size for recent heads")
    parser.add_argument("--wandb-project", type=str, default="search-head")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--generate-every", type=int, default=1, help="Generate sample every N epochs")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    tokens_per_epoch = args.tokens_per_epoch
    local_window = args.local_window

    # Ensure checkpoint directory exists
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_name or f"searchHead-byte-W{local_window}-N{N}-D{D}-L{N_LAYER}"

    # Recover wandb run id from checkpoint so we can resume the same run
    resume_run_id = None
    if args.resume and LATEST_CHECKPOINT_FILE.is_file():
        try:
            _ckpt_peek = torch.load(LATEST_CHECKPOINT_FILE, map_location="cpu", weights_only=False)
            resume_run_id = _ckpt_peek.get("wandb_run_id")
            del _ckpt_peek
        except Exception as e:
            print(f"Warning: could not read wandb_run_id from checkpoint: {e}")

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=resume_run_id,
        config={
            "version": "search-head-byte-v1",
            "N": N, "D": D, "N_HEAD": N_HEAD, "N_LAYER": N_LAYER,
            "W": local_window, "MLP_HEAD_HIDDEN": MLP_HEAD_HIDDEN,
            "HEAD_INPUT": f"2*D = {2*D}",
            "ALL_POS_WEIGHT": ALL_POS_WEIGHT,
            "LAST_POS_WEIGHT": LAST_POS_WEIGHT,
            "LR": LR, "WARMUP_STEPS": WARMUP_STEPS, "WEIGHT_DECAY": WEIGHT_DECAY,
            "BATCH_SIZE": batch_size, "EPOCHS": epochs,
            "TOKENS_PER_EPOCH": tokens_per_epoch,
            "VOCAB_SIZE": VOCAB_SIZE,
            "CODE_DATASET": CODE_DATASET_NAME,
        },
        resume="allow" if args.resume else None,
    )

    print("Loading FineWeb-Edu (streaming)...")
    hf_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    print("Loading CodeParrot (streaming)...")
    code_dataset = load_dataset(CODE_DATASET_NAME, split="train", streaming=True, revision="refs/convert/parquet")

    model = SearchHeadGPT(
        vocab_size=VOCAB_SIZE, n_embd=D, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=N, mlp_head_hidden=MLP_HEAD_HIDDEN, local_window=local_window,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | block_size={N} | head_input=2*{D} | local_window={local_window}")
    print(f"Heads: {N_HEAD // 2} recent (last {local_window} tokens) + {N_HEAD // 2} historical")
    print(f"Search head: tries all previous embeddings, picks max-confidence pair")
    print(f"Datasets: fineweb-edu (odd steps) + codeparrot (even steps)")
    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    optimizer = rebuild_optimizer(model, LR, WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best_val = float("inf")
    global_step = 0
    if args.resume and LATEST_CHECKPOINT_FILE.is_file():
        ckpt = torch.load(LATEST_CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt['epoch']} (lr={scheduler.get_last_lr()[0]:.2e})")
    elif args.load_weights:
        ckpt = torch.load(args.load_weights, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded weights from {args.load_weights} (starting from epoch 1)")

    print(f"\nTokens per epoch: {tokens_per_epoch:,}  |  Epochs: {epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}")
        epoch_seed = random.randint(0, 2**31)
        shuffled_ds = hf_dataset.shuffle(seed=epoch_seed, buffer_size=10_000)
        shuffled_code = code_dataset.shuffle(seed=epoch_seed, buffer_size=10_000)

        loader_kwargs = dict(
            batch_size=batch_size,
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        fineweb_loader = iter(DataLoader(
            TextDataset(shuffled_ds, seq_len=N, text_field="text"),
            **loader_kwargs,
        ))
        code_loader = iter(DataLoader(
            TextDataset(shuffled_code, seq_len=N, text_field=CODE_TEXT_FIELD),
            **loader_kwargs,
        ))

        model.train()

        total_loss_all = 0.0
        total_loss_last = 0.0
        total_tokens = 0
        total_batches_count = 0
        total_correct = 0
        total_correct_last = 0
        epoch_bytes = 0
        val_batches = []
        tic_epoch = time.time()
        step = 0
        all_best_j = []
        total_distance = 0.0
        total_distance_count = 0

        while True:
            step += 1
            try:
                if step % 2 == 1:
                    xb, yb = next(fineweb_loader)
                else:
                    xb, yb = next(code_loader)
            except StopIteration:
                break

            if step <= VAL_BATCHES:
                val_batches.append((xb.clone(), yb.clone()))
                continue

            global_step += 1
            if global_step <= WARMUP_STEPS:
                warmup_lr = LR * global_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            with torch.amp.autocast(DEVICE, enabled=use_amp):
                logits, loss, loss_all, loss_last, best_j = model(xb, targets=yb)

            with torch.no_grad():
                T = xb.size(1)
                query_positions = torch.arange(1, T, device=xb.device).unsqueeze(0).expand(xb.size(0), -1)
                distances = (query_positions - best_j).float()
                total_distance += distances.sum().item()
                total_distance_count += distances.numel()
                if step % 50 == 0:
                    all_best_j.append(best_j.cpu())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            valid_targets = yb[:, 1:]
            mask_start = max(0, LOSS_MASK_POSITIONS - 1)
            masked_targets = valid_targets[:, mask_start:]
            batch_tokens = masked_targets.numel()
            total_loss_all += loss_all.item() * batch_tokens
            total_loss_last += loss_last.item() * xb.size(0)
            total_tokens += batch_tokens
            total_batches_count += xb.size(0)
            masked_logits_train = logits[:, mask_start:, :]
            total_correct += (masked_logits_train.argmax(dim=-1) == masked_targets).sum().item()

            with torch.no_grad():
                total_correct_last += (logits[:, -1, :].argmax(dim=-1) == valid_targets[:, -1]).sum().item()

            epoch_bytes += xb.numel()

            if step % 50 == 0 or step <= 3:
                avg_all = total_loss_all / max(total_tokens, 1)
                avg_last = total_loss_last / max(total_batches_count, 1)
                acc = 100.0 * total_correct / max(total_tokens, 1)
                pct = epoch_bytes / tokens_per_epoch * 100
                print(
                    f"  step {step:5d}  "
                    f"all={avg_all:.4f} last={avg_last:.4f}  "
                    f"acc={acc:.1f}%  "
                    f"bpc={avg_all / math.log(2):.3f}  "
                    f"{pct:.0f}%",
                    end="\r", flush=True,
                )

            if epoch_bytes >= tokens_per_epoch:
                break

        scheduler.step()

        train_loss = total_loss_all / max(total_tokens, 1)
        train_loss_last = total_loss_last / max(total_batches_count, 1)
        train_acc = 100.0 * total_correct / max(total_tokens, 1)
        train_acc_last = 100.0 * total_correct_last / max(total_batches_count, 1)

        val_loss, val_last, val_acc, val_acc_last = evaluate(model, val_batches, use_amp)

        toc_epoch = time.time()
        print(
            f"\nEpoch {epoch:3d}  "
            f"train: all={train_loss:.4f} last={train_loss_last:.4f}  "
            f"acc={train_acc:.1f}% last_acc={train_acc_last:.1f}%  "
            f"val: all={val_loss:.4f} last={val_last:.4f}  "
            f"val_acc={val_acc:.1f}% last={val_acc_last:.1f}%  "
            f"bpc={val_loss / math.log(2):.3f}  "
            f"time={toc_epoch - tic_epoch:.0f}s",
            flush=True,
        )

        if epoch % args.generate_every == 0:
            prompt = b"The meaning of life is"
            sample = generate(model, list(prompt), max_new_tokens=150)
            sample_text = sample.decode("utf-8", errors="replace")
            print(f"  Sample: {sample_text[-200:]}")
            wandb.log({"sample": wandb.Html(f"<pre>{sample_text[-500:]}</pre>")}, commit=False)

        mean_distance = total_distance / max(total_distance_count, 1)
        search_log = {"search/mean_distance": mean_distance}
        if all_best_j:
            all_j_cat = torch.cat(all_best_j, dim=0).flatten()
            search_log["search/mean_abs_position"] = all_j_cat.float().mean().item()
            search_log["search/median_abs_position"] = all_j_cat.float().median().item()
            all_q = torch.arange(1, N, device="cpu").unsqueeze(0).expand(all_j_cat.size(0) // (N - 1), -1).flatten()
            if all_q.size(0) == all_j_cat.size(0):
                frac_prev = (all_j_cat == (all_q - 1)).float().mean().item()
                search_log["search/frac_t_minus_1"] = frac_prev
            if epoch % 5 == 0:
                search_log["search/position_histogram"] = wandb.Histogram(all_j_cat.numpy(), num_bins=64)

        wandb.log({
            "epoch": epoch,
            "train/loss_all": train_loss,
            "train/loss_last": train_loss_last,
            "train/bpc": train_loss / math.log(2),
            "train/bpc_last": train_loss_last / math.log(2),
            "train/acc": train_acc,
            "train/acc_last": train_acc_last,
            "val/loss_all": val_loss,
            "val/loss_last": val_last,
            "val/bpc": val_loss / math.log(2),
            "val/bpc_last": val_last / math.log(2),
            "val/acc": val_acc,
            "val/acc_last": val_acc_last,
            "lr": scheduler.get_last_lr()[0],
            **search_log,
        })

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "best_val": best_val,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        }, LATEST_CHECKPOINT_FILE)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "best_val": best_val,
                "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            }, CHECKPOINT_FILE)
            print(f"  -> Saved best (val_bpc={val_loss / math.log(2):.3f})")

    wandb.finish()


if __name__ == "__main__":
    main()
