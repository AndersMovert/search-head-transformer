# Search Head Transformer: Architecture Report

## 1. Core Mechanism: Confidence-Based Search Head

### Problem Statement

Standard transformers predict the next token using only the final hidden state `h[t]` at position `t`. This forces the entire context—local syntax, long-range dependencies, and factual recall—to be compressed into a single D-dimensional vector before the output layer makes its decision.

### Solution: Two-Phase Search

Instead of a linear projection `h[t] → logits`, the search head pairs `h[t]` with every previous embedding `h[j]` (j < t) and selects the pair that produces the most confident prediction.

**Phase 1 (Search, no gradient):**
- For each query position t, evaluate all candidate pairs (h[t], h[j]) through an MLP head
- Score each pair by the maximum probability in its output distribution (confidence)
- Select `best_j = argmax_j max_v P(v | h[t], h[j])`

**Phase 2 (Prediction, with gradient):**
- Recompute the selected pair (h[t], h[best_j]) with gradients enabled
- Use the resulting logits for loss computation and backpropagation

### Architecture

```
Input tokens → Embedding + Position → Transformer Backbone (N layers)
                                            ↓
                                    Hidden states h[0..T-1]
                                            ↓
                              Phase 1: Search all pairs (no grad)
                              Score: max(softmax(MLP(h[t] || h[j])))
                                            ↓
                              Phase 2: Selected pair (with grad)
                              Logits = MLP(h[t] || h[best_j])
```

**Head MLP:** `Linear(2D → hidden) → GELU → Linear(hidden → V)`

### Key Properties

1. **Non-degenerate search**: The model avoids trivially selecting `t-1` (the immediately preceding position). Empirically, `frac_t_minus_1` starts at ~1% (below uniform random of ~2.2%), confirming the search learns meaningful retrieval.

2. **Gradient flow**: Although Phase 1 is non-differentiable (argmax), Phase 2 passes gradients through both the backbone and the head for the selected pair. The backbone learns to produce embeddings that are *searchable*.

3. **Training signal**: The confidence-based selection creates a self-supervised curriculum—the model naturally finds pairs where the additional context most reduces uncertainty.

### Results

| Model | BPC at 2B chars | Architecture |
|-------|----------------|--------------|
| Search Head | 1.50 | h[t] + searched h[best_j] |
| Concat-K=5 | 1.85 | h[t] + last 5 embeddings |

The 0.35 BPC gap demonstrates that **selective context is dramatically more valuable than blind concatenation of recent context**. The search mechanism finds informative long-range dependencies that fixed-window approaches miss entirely.

---

## 2. Extension to External Buffers (RAG)

### From Internal to External Search

The search mechanism generalizes beyond the model's own context window. Any collection of embeddings—from any source—can serve as a search buffer:

```
Query h[t] ──→ Search buffer (any size) ──→ best match embedding
              ↓
         MLP head input: [h[t], best_match]
```

### Training Stages

**Critical constraint:** The ability to search external buffers must be learned in a dedicated post-training step. The training proceeds in two stages:

**Stage 1 — Pretraining (internal search only):**
- The model learns the basic search mechanism over its own context window
- The head learns to pair h[t] with informative h[j] from within the same sequence
- The backbone learns to produce embeddings that are searchable
- This is the stage demonstrated in current experiments (1.50 BPC at 2B chars)

**Stage 2 — Post-training (external buffer search):**
- The head is expanded from 2D input to (N_slots × D) input to accommodate additional buffer slots
- External buffers are introduced: embeddings from other documents, code files, etc.
- The stored embeddings in external buffers are typically the final hidden state from a separate forward pass (e.g., the last position's embedding from running a document through the same backbone)
- The model learns to search across these external embeddings alongside its own context
- The backbone may be frozen or fine-tuned with a lower learning rate during this stage

**Why this separation is necessary:**
- During pretraining, all candidate embeddings come from the *same forward pass*—they share positional encoding, attention context, and gradient flow. The model can exploit these correlations.
- External buffer embeddings come from *different forward passes* over different documents. They have different positional contexts and were computed independently.
- The head must learn that external embeddings represent a different "address space"—useful for factual/structural retrieval but lacking the local syntactic relationships present in internal context.
- Attempting to train external search from scratch alongside the base search mechanism would confuse the learning signal and slow convergence of both.

**Practical post-training setup:**
```
Training sample:
  - Input sequence: tokenized text (as in pretraining)
  - External buffer: embeddings from K other documents, computed offline
  - Head input: [h[t], best_internal, best_external_1, ..., best_external_N]
  - Loss: standard next-token prediction
```

### Multi-Buffer Architecture

The head input expands to include results from multiple independent searches, **performed in a fixed priority order:**

```
Head input: [h[t], best_local, best_rag, best_agent, best_human]
             D   +    D     +    D    +     D     +     D      = 5D

Search order (most to least important):
  1. Local context     — own sequence embeddings
  2. RAG documents     — codebase, docs, knowledge base
  3. External models   — other model outputs (agent buffer)
  4. Human input       — live user communication
```

**The search is autoregressive over buffers.** Each search sees the results of all prior searches — later searches are conditioned on richer context:

```
Step 1: score(h[t], 0,          0,        0,          candidate_local) → best_local
Step 2: score(h[t], best_local, 0,        0,          candidate_rag)   → best_rag
Step 3: score(h[t], best_local, best_rag, 0,          candidate_agent) → best_agent
Step 4: score(h[t], best_local, best_rag, best_agent, candidate_human) → best_human
```

This means **the search is NOT permutation invariant** — reversing the order would select entirely different items from every buffer. The ordering is architectural, baked into the learned weight blocks of the head MLP. Each slot position has its own learned projection weights, so the network implicitly encodes both source identity and source reliability per slot.

**Why this ordering:** Local context provides the syntactic and semantic grounding without which no prediction makes sense. RAG search, conditioned on what was found locally, can select domain-relevant knowledge. Agent search, seeing both local and RAG context, can provide targeted quality corrections. Human input, seeing everything else, serves as a final override. This mirrors a CPU's memory hierarchy (registers → L1 → L2 → L3 → RAM → disk) where the fastest, most critical data is accessed first.

| Priority | Buffer | Source | Update Rate | Purpose |
|----------|--------|--------|-------------|---------|
| 1 (highest) | Local context | Own sequence embeddings | Every token | Syntactic/semantic context |
| 2 | RAG documents | Codebase, docs, knowledge base | On file change | Domain knowledge retrieval |
| 3 | Agent buffer | Other model outputs | Continuous | Quality steering, verification |
| 4 (lowest) | Human buffer | User input during generation | Asynchronous | Intent correction, feedback |

### Scaling Properties

- **O(1) output width** regardless of buffer size: Whether the RAG buffer contains 100 or 10 million entries, the search returns exactly one D-dimensional embedding
- **O(n) search cost** per buffer (or O(log n) with approximate nearest neighbor indexing)
- **No context window consumption**: External information enters through the search slot, not by stuffing the sequence

### Comparison with Existing RAG

| Approach | Context cost | Blocking? | Granularity |
|----------|-------------|-----------|-------------|
| Naive RAG (prepend chunks) | O(k * chunk_size) | Yes (reformat prompt) | Document-level |
| Cross-attention RAG (RETRO) | O(k * D) attention | Yes (added layers) | Chunk-level |
| **Buffer search** | **O(1)** | **No** | **Embedding-level** |

---

## 3. Memory-Mapped I/O for Transformers

### The Analogy

In operating systems, memory-mapped I/O allows a CPU to communicate with hardware devices by reading/writing to fixed memory addresses. The CPU never blocks; devices update their registers independently at their own clock rates.

| OS Concept | Transformer Equivalent |
|---|---|
| CPU | Generation loop (token-by-token forward pass) |
| Device registers | Search buffers (local, RAG, agent, human) |
| Memory read instruction | Search query from h[t] into a buffer |
| Returned value | Best-matching embedding + confidence score |
| Polling rate | Once per generated token |
| Device update rate | Independent — human types whenever, agents write whenever |
| Address space layout | Fixed slot ordering in head input |
| Interrupt | Confidence exceeding threshold (future: could trigger early stop/redirect) |

### Why This Matters

**Non-blocking communication:** The model never stops generating to "check" its inputs. At every token, it searches all buffers as a natural part of its forward pass. If nothing relevant is found, the confidence-gated NULL embedding passes through and the model continues from its own context alone.

**Asynchronous input:** A human can type feedback ("too verbose", "wrong approach") into the human buffer *while the model is generating*. The model picks up the signal at the next token where it becomes the most relevant search result—no restart, no context stuffing, no explicit interruption protocol.

**Uniform interface:** The model doesn't need to know whether information comes from a file, a human, or another model. It uses the same learned search mechanism for all sources. Adding a new information source requires only adding a buffer and (optionally) training the head to use an additional input slot.

### Practical Applications

1. **Live coding feedback**: User flags "wrong approach" while model writes code → model self-corrects mid-function without regeneration
2. **Continuous environment updates**: File watcher detects changes → RAG buffer refreshes → model's next token reflects new state
3. **Multi-model orchestration**: Verifier writes "logical error at step 3" to agent buffer → reasoning model adjusts trajectory without stopping

---

## 4. Multi-Core Architecture

### From Single to Multi-Core

A single generation stream with multiple buffers is analogous to a single-core CPU with memory-mapped peripherals. The natural extension: **multiple generation streams (cores) sharing buffers and searching each other's contexts.**

### Architecture

For a 4-core system, each core searches in **strict priority order** (most to least important):

1. **Its own context** (local attention window) — highest priority
2. **The context of the other 3 cores** (3 separate peer searches)
3. **RAG buffer** (shared codebase/knowledge)
4. **External model buffer** (models on other machines)
5. **Human input buffer** (live user communication) — lowest priority

The ordering reflects information criticality: a core's own local context is essential for coherent generation, peer contexts provide collaborative signal, RAG provides knowledge, and external/human buffers provide steering corrections.

Per-core head input: `[h[t], best_self, best_core1, best_core2, best_core3, best_rag, best_ext, best_human]` = 8D

### Communication Topology

```
┌─────────────┐       search        ┌─────────────┐
│   Core 0    │◄────────────────────►│   Core 1    │
│   (coder)   │                      │  (reviewer) │
└──────┬──────┘                      └──────┬──────┘
       │ search                              │ search
       ▼                                     ▼
┌─────────────┐       search        ┌─────────────┐
│   Core 2    │◄────────────────────►│   Core 3    │
│  (tester)   │                      │  (planner)  │
└─────────────┘                      └─────────────┘
       ▲              ▲              ▲
       │              │              │
       └───── shared buffers ────────┘
            (RAG, external, human)
```

### Key Properties

**Latent-state communication:** Each core accesses the *hidden states* of peer cores, not their output text. This is richer than token-level communication and available before a peer finishes generating. Core 1 (reviewer) can reason about issues in Core 0's (coder) code as it emerges.

**Non-blocking:** No core waits for another. Each searches peer buffers at its own generation rate. If a peer hasn't produced relevant context yet, the search returns low-confidence and the NULL embedding passes through.

**Scaling:**

| Cores | Searches per token per core | Total searches/token |
|-------|---------------------------|---------------------|
| 1 | 4 (self + rag + ext + human) | 4 |
| 2 | 5 + 1 peer | 12 |
| 4 | 4 + 3 peers | 32 |
| 8 | 4 + 7 peers | 96 |

### Cross-Model Communication (Different Weights)

Peer cores that share the same weights can exchange raw hidden states directly — the embeddings are natively compatible. But communication with **foreign models** (different architecture, different weights, different vocabulary) requires a text gateway:

```
Foreign model (e.g., GPT, Claude, Gemini)
         │
         ▼ generates text
   "The function should handle edge case X..."
         │
         ▼ received over network
   Your backbone forward pass (frozen, no grad)
         │
         ▼ produces native embeddings
   h[0], h[1], ..., h[N]  ← now searchable in your embedding space
         │
         ▼ inserted into external model buffer
   Search head queries this buffer like any other
```

**The key insight:** Any model that produces text can participate in the buffer system. You receive their text, run it through your own backbone, and the resulting embeddings are native to your search space. No shared weights, no shared architecture, no shared vocabulary required.

| Communication mode | Same weights | Different weights |
|---|---|---|
| **Signal** | Raw hidden states h[j] | Text → re-encode through own backbone → h[j] |
| **Richness** | Full latent representation | Re-encoded (lossy but compatible) |
| **Latency** | Per-token (stream hidden states) | Per-message (wait for text, then encode) |
| **Use case** | Multi-core peers | Cross-model (remote APIs, different architectures) |

**Graceful degradation:** The system supports a spectrum of communication fidelity. Same-weights peers get the richest channel (per-token hidden state streaming). Foreign models get a slightly lossy but fully functional channel through text re-encoding. Both use the same search mechanism — the head doesn't need to know how the buffer was populated.

### Applications

1. **Parallel decomposition**: Core 0 writes code, Core 1 writes tests, Core 2 writes documentation—all synchronized through buffer search rather than explicit orchestration
2. **Speculative execution**: Fast draft core generates tokens; verification core checks via buffer search and signals corrections
3. **Distributed inference**: Cores on different machines communicate through shared embedding buffers over the network—no custom protocol needed
4. **Cross-architecture collaboration**: A local coding model searches embeddings re-encoded from a remote reasoning model's text output — combining specialized capabilities without distillation or fine-tuning
5. **Reasoning with dense feedback**: During chain-of-thought, a critic core searches the reasoning core's hidden states and writes verdicts to the agent buffer, providing continuous steering

### Comparison with Existing Multi-Agent Systems

| Approach | Communication | Blocking? | Granularity | Requires orchestrator? |
|----------|--------------|-----------|-------------|----------------------|
| LangChain agents | Text messages | Yes | Full responses | Yes |
| CrewAI | Turn-taking | Yes | Task-level | Yes |
| Debate (Irving et al.) | Alternating text | Yes | Argument-level | Yes |
| Tree-of-Thought | Branch-and-evaluate | Yes | Step-level | Yes |
| **Multi-core search** | **Embedding search** | **No** | **Per-token** | **No** |

---

## 5. Compute Economics

### The Chinchilla Argument for Search Heads

The search head compresses the "easy pattern" learning phase, allocating more of the training budget to high-value late-stage learning:

```
Vanilla:     [======= easy patterns =======][=== hard patterns ===]
              0B ──────────────── 10B ────────────── 20B

Search head: [== easy ==][============ hard patterns ============]
              0B ── 2B ────────────────── 20B
```

At a fixed 20B training budget:
- Vanilla spends ~10B reaching search head's 2B-performance, then 10B on hard patterns
- Search head reaches the same point at 2B, spends 18B on hard patterns
- **~80% more compute on the high-value phase**

### Token-Level Vocab Amplification

With larger vocabularies (BPE 4096–128k), the softmax decision is harder. The search embedding acts as a sharpening signal:
- V=256 (bytes): Search provides 0.35 BPC gain
- V=4096+ (BPE): Expected gain scales with log₂(V) — potentially 0.5–1.0+ bits/token
- V=128k (Llama): Maximum potential impact—largest decision space to narrow

### Frozen Backbone + Search Head

The search head can be trained independently on top of a frozen pretrained model:
- Only train ~5-10% of total params (the MLP head)
- Backbone forward pass needed but no backward through it
- Single 24GB GPU sufficient with gradient checkpointing
- Applicable to any existing model (Llama, Phi, Mistral)

---

## 6. Summary

The search head transforms the transformer from a **batch sequence processor** into a **reactive system with non-blocking read access to an evolving information environment**:

| Layer | Concept | Enables |
|-------|---------|---------|
| Base | Confidence-based pair search | Better next-token prediction |
| +RAG | External buffer search | Unlimited knowledge without context stuffing |
| +MMIO | Async input buffers | Live human/agent feedback during generation |
| +Multi-core | Peer context search | Parallel specialized generation with implicit coordination |

The fundamental insight: **a learned search mechanism is a universal communication primitive**. Any information source—internal context, files, humans, other models, remote systems—can participate by providing embeddings in a shared space. The model learns what to retrieve, when, through end-to-end training.

This is not an incremental improvement to transformers. It is a new inference paradigm.
