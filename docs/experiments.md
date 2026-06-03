# Experiments

Planned and in-progress experiments for the search head transformer.

---

## 1. Search Head vs Concat-K=5 Baseline

**Hypothesis:** A learned search over all previous embeddings outperforms blind concatenation of the last K=5 embeddings as input to the output head.

**Background:** The initial idea was to concatenate the K last embeddings and train to convergence, then add search as a post-training step (search over all remaining embeddings in the context, pick the one producing the highest probability when concatenated with the other K embeddings). While waiting for this experiment to finish, the idea came up to learn to search as part of pretraining — so the comparison became: concat-K=5 head vs search head, both trained from scratch.

**Setup:**
- Concat-K=5: `MLP([h[t], h[t-1], h[t-2], h[t-3], h[t-4]]) → logits`
- Search head: `MLP([h[t], h[best_j]]) → logits` where `best_j = argmax confidence`
- Same backbone, same data, same training schedule

**Status:** Running

You can view the live training metrics on [Weights & Biases](https://api.wandb.ai/links/b5dr6zq2qt-n-a/4lcgkept)

**Results so far:** The search head transformer performs much better than blind concatenation of the last K=5 embeddings.


---

## 2. Search Head vs Vanilla Transformer

**Hypothesis:** The search head converges faster and to a higher accuracy than a vanilla transformer (standard `Linear(h[t]) → logits`).

**Background:** The search head appears to compress the "easy pattern" learning phase, reaching the same loss level in fewer tokens. This experiment will quantify the gap and determine whether it's a convergence speed advantage or a permanent quality gap.

**Setup:**
- Vanilla: `Linear(h[t]) → logits` (standard single-embedding output head)
- Search head: `MLP([h[t], h[best_j]]) → logits`
- Same backbone, same data, same training schedule

**Status:** Running

You can view the live training metrics on [Weights & Biases](https://api.wandb.ai/links/b5dr6zq2qt-n-a/vbvaty6p)

**Results so far:** The search head transformer does not converge faster compared to a vanilla transformer. But will it converge to a higher accuracy? Stay tuned.

---

## 3. Post-Training with External Buffer (size 256)

**Hypothesis:** Adding an external embedding buffer during post-training improves accuracy beyond what internal search alone achieves.

**Background:** To test external buffer search, we extend the training data: instead of fetching 256+1 characters per batch member, we fetch 3×256+1 characters. The first 2×256 characters are run through the frozen backbone to produce 256 embeddings that populate the external buffer. The final 256+1 characters are used for the actual training sequence.

**Setup:**
- Freeze all pretrained search head weights
- Expand the head's first linear layer with additional columns to accept a third embedding: `MLP([h[t], best_internal, best_external]) → logits`
- Only the new parameters are trained
- External buffer: 256 embeddings from preceding context (same document)
- Loss: standard next-token prediction

**Status:** Not started

***Plot and table of results go here***

---

## 4. Pre-Training with External Buffer (size 256)

**Hypothesis:** It is better to learn to search both internal and external buffers during pre-training, than post-train with external buffer.

**Background:** The external buffer has a different positional encoding than the internal buffer. The first position in the context window has a very large error in both the vanilla transformer and the search head transformer, but pre-training with an external buffer will make it will make it possible to push down the errors on the first position in the context window, which will probably drive the training of the external buffer search.

**Setup:**
- Train from scratch — no pretrained weights, no frozen components
- Single unified head: `MLP([h[t], best_internal, best_external]) → logits`
- Internal search: score `[h[t], h[j], zeros]` through head weights for all j < t, pick highest confidence
- External search: score `[h[t], best_internal, buf[k]]` through same head weights for all k in buffer
- All parameters updated end-to-end (backbone + head)
- External buffer: 256 embeddings from preceding context (sliding window, stride 1, last-position embedding)
- Loss: standard next-token prediction
- Script: `src/search_head_ext_buffer_pretrain.py`

**Status:** Not started

***Plot and table of results go here***

---

## 5. External Buffer Generalization (size 256 → 512)

**Hypothesis:** A model post-trained on a size-256 external buffer generalizes to larger buffers at inference time without additional training.

**Background:** If the search mechanism truly learns a general retrieval strategy (not just memorizing positional patterns within a fixed buffer size), it should benefit from more candidates at test time. This is analogous to how a CPU benefits from more RAM without needing hardware changes.

**Setup:**
- Freeze all parameters (including post-trained external search weights)
- Increase external buffer from 256 to 512 embeddings at inference
- Measure accuracy on the same validation set with both buffer sizes

**Status:** Not started

***Plot and table of results go here***

---

## 6. Pre-trained Search Head vs Post-trained Vanilla Transformer

**Hypothesis:** Post train a pre-trained vanilla transformer with a search head will get the same performance.

**Background:** If you can just add a search head to a pre-trained vanilla transformer and get the same gain, that would be very useful as a way to take large open-source models and post-train them with a search head. The answer is likely no, since when you pre-train with the search head, the embeddings are optimized end-to-end to make search effective — a vanilla backbone never received that gradient signal.

**Setup:**
- Model A (baseline): Search head pre-trained from scratch (from Experiment 2)
- Model B (test): Take the converged vanilla transformer, freeze backbone, replace `Linear(h[t])` with `MLP([h[t], h[best_j]])`, post-train only the new search head parameters
- Same total compute budget for Model B's post-training as Model A's full training
- Compare final BPC on the same validation set

**Status:** Not started

***Plot and table of results go here***

---

## 7. TBD

Future experiments to consider:
- BPE-4096 vs byte-level search head gains
- Frozen pretrained backbone (Llama/Phi) + search head
- Multi-buffer with multiple external sources
- Ablation on search order (permutation sensitivity)
