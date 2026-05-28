# Learned Engrams: Attention-Derived Conditional Memory for a 300M Telugu LM

## Overview

This document describes **Learned Engrams** — a conditional memory mechanism for our 300M Telugu language model, inspired by DeepSeek's Engram module but with a fundamental difference: instead of deterministic N-gram lookup on raw token IDs, our patterns are **learned from the attention signal** and regularized via Minimum Description Length (MDL) to retain only structurally significant token patterns.

The core hypothesis: a small model wastes significant compute reconstructing local stereotyped patterns (morphological compounds, postposition phrases, named entities, formulaic collocations) that recur throughout Telugu text. By offloading these to an explicit memory with O(1) lookup, we free the transformer's limited depth for compositional reasoning — effectively deepening the network without adding layers.

---

## 1. Mechanism in Detail

### 1.1 Pattern Extraction: Attention-Derived Masked Spans

At each position `p` in the sequence, we examine a **contiguous window** of the `W` immediately preceding tokens: `[x_{p-W}, ..., x_{p-1}]`. We fix `W = 6`.

The QK attention logits from the current layer (averaged across heads) provide a signal for which positions in this window are "structurally salient" to the prediction at position `p`. Rather than using these logits directly as a threshold, we feed them into a **pattern predictor** — a small learned head (2-layer MLP, `W → 2W → W`) that outputs mask logits for each window position.

Each window position gets a binary decision: **retain** (mask=1, the token is structurally important) or **mask** (mask=0, the token is ignorable). The result is a **masked span** — e.g., for the window `[తెలుగు, లో, భాష, యొక్క, గొప్ప, చరిత్ర]`, the pattern might be `[తెలుగు, ∅, భాష, ∅, ∅, చరిత్ర]`.

**Gumbel-sigmoid sampling** produces differentiable hard binary masks during training (straight-through estimator: hard {0,1} in forward, smooth gradients in backward). At inference, masks are deterministic (sigmoid > 0.5).

### 1.2 MDL Regularization

Without constraint, the pattern predictor could retain all tokens (mask=1 everywhere), producing maximally specific but non-reusable patterns. The **Minimum Description Length** principle penalizes this over-specification.

We impose a sparse Bernoulli prior π = 0.3 (expecting ~2 of 6 positions unmasked on average) and compute the KL divergence:

```
L_MDL = λ · mean_over_positions[ Σ_w KL(Bernoulli(p_w) || Bernoulli(π)) ]
```

where `p_w` is the sigmoid probability of unmasking position `w`. This loss is added to the standard language modeling loss. The effect: the model retains a token in the span only when the information gain justifies the description cost. This produces sparse, recurring patterns that achieve high cache hit rates.

λ = 0.01 (tunable). π = 0.3 targets ~2 retained tokens per 6-position window. Lower π → sparser patterns → more cache reuse but less expressive; higher π → richer patterns → fewer cache hits.

### 1.3 Memory Table: Fixed-Size Hash Table

The memory is a **fixed-size embedding table** of `M = 1,000,000` slots, each storing a 128-dimensional span embedding vector.

**Total memory parameters:** 1M × 128 = **128M parameters**, stored in CPU RAM (not GPU).

**Addressing.** Once the pattern predictor produces a hard mask, the masked span is fully deterministic — same tokens in same positions with same mask always yields the same entry. The key for each span is a 6-tuple where retained positions carry their `token_id` and masked positions carry a **per-position sentinel value** (not zero):

```
key = (s_1, s_2, s_3, s_4, s_5, s_6)
     where s_i = token_id_i  if mask_i = 1 (retained)
           s_i = SENTINEL_i  if mask_i = 0 (masked)

SENTINEL_i = vocab_size + i + 1    (e.g., 48001, 48002, ..., 48006)
```

**Why not zero?** A zero sentinel annihilates the hash term at that position, reducing effective hash entropy. With sparse masks (3-4 of 6 positions masked), most terms contribute nothing, causing degenerate collision rates. Per-position sentinels above the vocab range ensure every position always contributes a non-zero, position-unique term to the hash — whether it holds a real token or a mask marker. The hash maintains full entropy regardless of mask sparsity.

This tuple is hashed to a table index via **multi-head multiplicative-XOR hashing** (following Engram):

- K = 4 independent hash heads per lookup
- Each head k uses a different hash function φ_k mapping the 6-tuple to an index in a prime-sized sub-table
- The K retrieved embeddings are concatenated to form the full 128-dim span embedding (each head contributes 128/K = 32 dims)
- Table sub-sizes: 4 prime-sized tables, each with ~250K slots

This multi-head scheme mitigates hash collisions. High-frequency patterns get clean dedicated embeddings; rare-tail patterns may collide but also contribute less to overall loss.

### 1.4 Fusion: Gated Additive Residual (from Engram)

We adopt Engram's proven fusion strategy directly. The retrieved span embedding `e_t` (128-dim) is fused with the transformer's hidden state `h_t` (1024-dim) as follows:

**Step 1 — Project to Key and Value:**

```
k_t = W_K · e_t     (128 → 1024)
v_t = W_V · e_t     (128 → 1024)
```

W_K and W_V are learnable projection matrices.

**Step 2 — Context-aware scalar gate:**

```
α_t = σ( RMSNorm(h_t)ᵀ · RMSNorm(k_t) / √d )
```

This gate ∈ (0, 1) measures semantic alignment between the current context (h_t, which has already seen global context via preceding attention layers) and the retrieved memory. If the memory contradicts or is irrelevant to the current context, α_t → 0 and the memory is suppressed.

**Step 3 — Gated value with causal convolution:**

```
v̂_t = α_t · v_t

Y = SiLU(Conv1D(RMSNorm(V̂))) + V̂
```

A lightweight depthwise causal convolution (kernel_size=4, dilation=6 matching our window size) with SiLU activation smooths the gated memory signal across adjacent positions. The conv is initialized to zero so the module starts as identity.

**Step 4 — Residual addition:**

```
H^(ℓ) ← H^(ℓ) + Y
```

The output is added to the hidden state before the standard Attention and MLP blocks at that layer. Pure additive — no modification to the self-attention mechanism.

### 1.5 Layer Placement

Following Engram's findings that early injection offloads local pattern reconstruction while later injection benefits from richer contextual gating:

For our target 28-32 layer model:
- **Layer 2**: Early intervention, offloads local Telugu morphological patterns before the backbone spends depth reconstructing them
- **Layer 15-16** (roughly mid-depth): Benefits from richer h_t for more precise gating decisions

Each injection point gets its own set of projection matrices (W_K, W_V) and conv parameters, but shares the same underlying memory table. The pattern predictor is also per-layer (since QK patterns differ by depth).

Total parameter overhead on GPU (excluding the memory table):
- 2 × pattern predictor: 2 × (6×12 + 12×6) = ~288 params (negligible)
- 2 × W_K projection: 2 × (128 × 1024) = ~262K params
- 2 × W_V projection: 2 × (128 × 1024) = ~262K params
- 2 × Conv1D (depthwise, kernel 4): 2 × (1024 × 4) = ~8K params
- Total on-GPU overhead: **~533K parameters** (0.18% of backbone)

---

## 2. System Design

### 2.1 Memory Residency

| Component | Location | Size |
|-----------|----------|------|
| Backbone (300M) | GPU | ~1.2 GB (bf16) |
| Memory table (1M × 128) | CPU RAM | ~512 MB (fp32) |
| Active rows per batch | GPU (transient) | ~8 MB |

The memory table is too large for GPU but trivially fits in CPU RAM. Per-batch, only the rows actually looked up are transferred.

### 2.2 Forward Pass Data Flow

```
Input token IDs (B, T)
        │
        ▼
   Vocab Embedding ──────────────────────────────────┐
        │                                            │
        ▼                                            │
   [Layer 0: Attention + MLP]                        │
        │                                            │
        ▼                                            │
   [Layer 1: Attention + MLP]                        │
        │                                            │
        ▼                                            │
   [Layer 2: LEARNED ENGRAM INJECTION]               │
        │                                            │
        │  ┌─────────────────────────────────┐       │
        │  │ For each position p:            │       │
        │  │  1. Extract QK window logits    │       │
        │  │  2. Pattern predictor → mask    │       │
        │  │  3. Gumbel-sigmoid → hard mask  │       │
        │  │  4. Form key: (tok×mask) tuple  │       │
        │  │  5. Hash → table index          │◄──────┘ (uses token IDs)
        │  │  6. Gather rows from CPU table  │
        │  │  7. Project e_t → k_t, v_t     │
        │  │  8. Gate: α = σ(h·k/√d)        │◄── h_t from backbone
        │  │  9. Conv + residual add         │
        │  └─────────────────────────────────┘
        │
        ▼
   [Layers 3-14: Attention + MLP]
        │
        ▼
   [Layer 15: LEARNED ENGRAM INJECTION (same table, own projections)]
        │
        ▼
   [Layers 16-27: Attention + MLP]
        │
        ▼
   LM Head → logits
```

### 2.3 CPU↔GPU Transfer

For a batch of 8 sequences × 2048 tokens = 16,384 positions:
- Each position produces one hash lookup (one 6-tuple → 4 hash heads → 4 sub-table lookups)
- Unique rows per batch (after dedup): typically 4K-10K due to pattern repetition within and across sequences
- Transfer size: 10K rows × 128 dims × 4 bytes = **~5 MB per injection layer**
- PCIe 4.0 bandwidth: ~25 GB/s → transfer time: **~0.2 ms**
- This can be overlapped with computation of the preceding transformer block (Engram's prefetch strategy)

### 2.4 Backward Pass

Gradients flow back through:
1. **Fusion parameters** (W_K, W_V, Conv1D) — standard backprop, on GPU
2. **Pattern predictor** — gradients through Gumbel-sigmoid (straight-through estimator) back to the MLP
3. **Memory table rows** — sparse gradient update to the rows that were retrieved

For the memory table:
- Use `torch.nn.Embedding` with `sparse=True`, placed on CPU
- Forward: `index_select` gathers rows to GPU
- Backward: sparse gradients are scattered back to CPU rows via `index_copy_`
- Optimizer: Adam with **5× the base learning rate** and **no weight decay** on the table (following Engram), since each row sees far fewer updates than backbone parameters

---

## 3. Training Protocol

### 3.1 Initialization

- **Memory table**: Zero-initialized. All 1M rows start at zero. Since the gate and conv are also zero-init, the module contributes nothing at the start of training, letting the backbone stabilize first.
- **Conv parameters**: Zero-initialized (identity mapping at start)
- **Gate**: Not a separate learnable scalar like in our earlier prototype. Instead, the context-aware gate α_t from the dot-product naturally starts near 0.5 (random hidden states vs zero memory keys → near-orthogonal → sigmoid(~0) ≈ 0.5). As the table learns meaningful embeddings, gates will sharpen.
- **Pattern predictor**: Standard initialization. Early in training, mask predictions are near-random. This is fine — the MDL loss will guide it toward sparse patterns, and the table will wash out early garbage entries as meaningful patterns emerge and dominate gradient signal.
- **W_K, W_V projections**: Xavier initialization

### 3.2 Warmup Considerations

No explicit warmup phase is needed for the memory module. The combination of:
- Zero-init table (no contribution until rows receive gradient)
- MDL loss (pushing toward sparse, high-reuse patterns)
- Context-aware gating (suppressing irrelevant retrievals)

...provides implicit warmup. The backbone trains normally for the first few hundred steps while the pattern predictor and table co-adapt.

However, we recommend **freezing the memory table for the first 1-2K steps** and only training the pattern predictor and fusion parameters. This lets the pattern predictor learn meaningful masks before the table starts accumulating entries. After unfreezing, the table fills rapidly with the (now meaningful) patterns.

### 3.3 Learning Rates

| Component | LR Multiplier | Weight Decay |
|-----------|---------------|-------------|
| Backbone (attention, MLP, norms) | 1× (base LR) | 0.1 |
| Pattern predictor | 1× | 0.1 |
| W_K, W_V projections | 1× | 0.1 |
| Conv1D | 1× | 0.0 |
| Memory table (CPU) | 5× | 0.0 |

The 5× multiplier on the memory table follows Engram's finding that embedding rows need faster learning due to sparse updates. No weight decay on the table because L2 regularization would shrink rarely-accessed rows toward zero, destroying stored knowledge.

### 3.4 Training Loss

```
L_total = L_LM + L_MDL_layer2 + L_MDL_layer15
```

where L_LM is the standard causal language modeling cross-entropy loss, and each L_MDL is the KL-based MDL penalty from the pattern predictor at that injection layer.

### 3.5 Gumbel Temperature Annealing

The Gumbel-sigmoid temperature τ controls mask hardness:
- τ = 1.0 at start (soft masks, easy gradient flow)
- Anneal to τ = 0.3 over the first 30% of training (hard masks, clean hash keys)
- Hold at τ = 0.3 for remainder

Harder masks are important for cache coherence — soft masks mean the same pattern maps to slightly different hash keys across forward passes, fragmenting the table.

---

## 4. Inference

At inference time:
1. The memory table is **frozen** (no updates)
2. The pattern predictor runs deterministically (no Gumbel noise, just sigmoid > 0.5)
3. Hash lookup is O(1) — compute the key, hash it, fetch the row
4. The table can be fully offloaded to CPU RAM with async prefetch (Engram's strategy)
5. Since hash indices are deterministic from token IDs alone, they can be **precomputed** before the forward pass reaches the injection layer, overlapping memory access with computation

**Latency impact:** Near zero. The 5 MB transfer per injection layer completes in ~0.2 ms, which is masked by the computation of the preceding transformer block.

---

## 5. Why This Differs From (and Complements) Engram

| Aspect | DeepSeek Engram | Learned Engrams (Ours) |
|--------|----------------|----------------------|
| **Pattern source** | Raw token ID N-grams (deterministic from input) | Attention-derived masked spans (learned) |
| **What's stored** | Static embedding per N-gram | Static embedding per masked span |
| **Pattern selection** | All N-grams at all positions | Only MDL-justified patterns (sparse) |
| **Mask/selection** | No masking — full contiguous N-gram | Binary mask selects structurally important tokens |
| **Window size** | 2-3 tokens (bigram, trigram) | 6 tokens with sparse mask (~2-3 retained) |
| **Effective span** | Very local (2-3 adjacent tokens) | Wider receptive field (6 tokens, non-contiguous retention) |
| **Fusion** | Same: gated additive residual + conv | Same: gated additive residual + conv |
| **Table size** | 5.7B params (at 27B backbone scale) | 128M params (at 300M backbone scale) |
| **Scale context** | Complement to MoE sparse params | Standalone augmentation to dense model |

The key advantage of learned masking: a 6-token window with 2 retained tokens captures patterns that would require a 6-gram in Engram's formulation. But the space of 6-grams over a 48K vocabulary is astronomically larger than the space of masked 6-grams with only 2-3 retained positions. The MDL loss ensures the model discovers the *structurally minimal* representation of each pattern.

For Telugu specifically, this is powerful. Agglutinative morphology means that the structurally important parts of a phrase are often non-adjacent: the stem and the case marker matter, the agglutinated suffixes between them are predictable from context. A masked span naturally captures `[stem, ∅, ∅, case_marker, ∅, ∅]` — something a contiguous bigram/trigram cannot represent.

---

## 6. Relationship to Existing Codebase

### What changes in `train_gpt.py`:

1. **CausalSelfAttention**: Add `return_qk` flag to expose raw QK logits (before softmax) at injection layers only. At non-injection layers, attention runs unchanged with Flash Attention.

2. **Block**: At injection layers (2 and 15), add the Learned Engram module. The block forward becomes:
   ```
   attn_out, qk_logits = self.attn(self.ln_1(x), return_qk=True)
   engram_out = self.engram(x, qk_logits, token_ids, tok_embeddings)
   x = x + attn_out + engram_out
   x = x + self.mlp(self.ln_2(x))
   ```

3. **GPT model**: Thread `token_ids` and `tok_embeddings` through the forward pass (tok_embeddings is already computed at the embedding layer; token_ids are the input).

4. **Training loop**: Add MDL loss to total loss. Set up separate optimizer param group for the memory table (5× LR, no weight decay, sparse Adam).

5. **Memory table**: Initialize as `nn.Embedding(1_000_000, 128, sparse=True)` on CPU. Manage CPU↔GPU transfers in the forward/backward hooks.

### What does NOT change:

- Architecture of non-injection layers (the vast majority)
- Tokenizer (Morfessor-based, whatever vocab size we settle on)
- Data pipeline
- Evaluation
- Checkpointing (memory table is saved/loaded alongside backbone)

---

## 7. Open Questions and Future Directions

1. **Optimal mask sparsity (π):** We set π=0.3 (~2 of 6 retained). This should be tuned. Lower π means sparser patterns, higher cache reuse, less expressive. A sweep over {0.15, 0.3, 0.45} on a smaller proxy model would inform this.

2. **Window size:** Fixed at 6. Could be worth testing 4 and 8. Smaller windows → denser patterns → more cache hits but less expressive. Telugu's morphological chunks tend to be 3-5 morphemes, suggesting 6 is reasonable.

3. **Number of injection layers:** We start with 2 (layers 2 and 15). Engram found diminishing returns beyond 2 injection points at their scale. At 300M, even a single injection at layer 2 may suffice, with the second providing incremental gains.

4. **Table growth over training:** The table starts empty and fills as training progresses. With 1M slots and Zipfian pattern distribution, we expect ~60-70% occupancy at convergence, with the top 10% of patterns accounting for ~50% of all lookups. Monitoring occupancy and collision rates during training will be important.

5. **Cross-epoch pattern stability:** As the pattern predictor evolves over training, patterns that were frequent early may become rare later (and vice versa). The table needs to handle this gracefully. Since we use gradient-based updates (not EMA), stale entries naturally decay as they stop receiving gradient signal. Active entries stay fresh.

6. **Interaction with tokenizer changes:** If we reduce vocab from 80K+ to 32-48K, the pattern space changes. The memory module should be designed after the tokenizer is finalized, or retrained from scratch if the tokenizer changes.

7. **QK logit availability at injection layers:** At layer 2, the QK logits are based on very early representations (just 2 layers of processing). These may not carry strong attention signal yet. However, Engram found that layer 2 injection works well precisely because the task there is local pattern reconstruction, which doesn't require deep contextual understanding. The pattern predictor can learn to extract useful signal even from shallow QK patterns.

---

## 8. Summary

Learned Engrams combine three ideas:
1. **From Engram:** Gated additive residual fusion with causal convolution, layer-sparse injection, CPU-offloaded memory table, multi-head hashing
2. **Novel — attention-derived masking:** Instead of deterministic N-gram keys, use a learned pattern predictor on QK logits to decide which tokens in a local window are structurally important
3. **Novel — MDL regularization:** Penalize over-specification of patterns, ensuring the model discovers the minimal description of each recurring local structure

The result: a conditional memory that is more expressive than fixed N-gram lookup (captures non-contiguous structural patterns in agglutinative Telugu) while retaining O(1) deterministic lookup at inference. The 128M parameter table lives off-GPU, adding ~0.18% parameter overhead on-GPU and near-zero latency overhead.
