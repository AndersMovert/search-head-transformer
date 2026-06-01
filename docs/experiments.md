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

***Plot and table of results go here***

---

## 2. Search Head vs Vanilla Transformer

**Hypothesis:** The search head converges faster and to a higher accuracy than a vanilla transformer (standard `Linear(h[t]) → logits`).

**Background:** The search head appears to compress the "easy pattern" learning phase, reaching the same loss level in fewer tokens. This experiment will quantify the gap and determine whether it's a convergence speed advantage or a permanent quality gap.

**Setup:**
- Vanilla: `Linear(h[t]) → logits` (standard single-embedding output head)
- Search head: `MLP([h[t], h[best_j]]) → logits`
- Same backbone, same data, same training schedule

**Status:** Not started

***Plot and table of results go here***

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

## 4. External Buffer Generalization (size 256 → 512)

**Hypothesis:** A model post-trained on a size-256 external buffer generalizes to larger buffers at inference time without additional training.

**Background:** If the search mechanism truly learns a general retrieval strategy (not just memorizing positional patterns within a fixed buffer size), it should benefit from more candidates at test time. This is analogous to how a CPU benefits from more RAM without needing hardware changes.

**Setup:**
- Freeze all parameters (including post-trained external search weights)
- Increase external buffer from 256 to 512 embeddings at inference
- Measure accuracy on the same validation set with both buffer sizes

**Status:** Not started

***Plot and table of results go here***

---

## 5. TBD

Future experiments to consider:
- BPE-4096 vs byte-level search head gains
- Frozen pretrained backbone (Llama/Phi) + search head
- Multi-buffer with multiple external sources
- Ablation on search order (permutation sensitivity)
