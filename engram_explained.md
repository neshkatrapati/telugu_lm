# Engram: Conditional Memory via Scalable Lookup — A Detailed Explanation

**Paper**: *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*
**Authors**: Xin Cheng, Wangding Zeng, Damai Dai, et al. (Peking University & DeepSeek-AI)
**Code**: [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)

---

## TL;DR

Current LLMs use the same expensive neural computation for everything — both complex reasoning *and* simple pattern recall (like recognizing "Alexander the Great" or "by the way"). Engram adds a **lookup table** alongside the neural network, so the model can cheaply *look up* static knowledge patterns via O(1) hashing instead of burning layers of attention+MLP to reconstruct them. This frees up the neural network's depth for actual reasoning. The result: a 27B model that beats an iso-parameter, iso-FLOPs MoE baseline across knowledge, reasoning, code, math, *and* long-context tasks.

---

## 1. The Core Problem: LLMs Waste Depth on Memorization

Language modeling involves two fundamentally different sub-tasks:

1. **Compositional reasoning** — understanding context, drawing inferences, multi-step logic
2. **Knowledge retrieval** — recalling named entities, idioms, formulaic phrases

These two tasks have very different computational requirements. Reasoning requires deep, dynamic computation. But knowledge retrieval is essentially a *lookup* — the answer is static and doesn't depend on complex reasoning.

**The problem**: Current Transformers have no native lookup mechanism. They're forced to *simulate* retrieval through layers of attention and feed-forward networks. For example, to recognize the entity "Diana, Princess of Wales", an LLM progressively composes features across 6+ layers:

| Layer | What the model "thinks" the last token means |
|-------|----------------------------------------------|
| 1-2   | "Wales" = Country in the United Kingdom |
| 3     | "Wales" = Country in Europe |
| 4     | "Princess of Wales" = Title held by female sovereigns |
| 5     | "Princess of Wales" = Title given to wife of Prince of Wales |
| 6     | "Diana, Princess of Wales" = Diana (1961-1997), first wife of Prince Charles |

This burns 6 layers of expensive computation to reconstruct what could be a simple table lookup. Those layers could have been doing *reasoning* instead.

---

## 2. The Key Idea: Conditional Memory as a New Sparsity Axis

The paper proposes adding a second kind of sparsity alongside MoE (Mixture of Experts):

| Paradigm | What it does | How it scales |
|----------|-------------|---------------|
| **Conditional Computation (MoE)** | Sparsely activates *neural parameters* for dynamic processing | More experts = more capacity, but more FLOPs per active expert |
| **Conditional Memory (Engram)** | Sparsely *looks up* static embeddings via hashing | More table entries = more knowledge, with O(1) cost per lookup |

The insight: **MoE and Engram are complementary**. MoE handles dynamic reasoning. Engram handles static knowledge. Neither alone is optimal.

---

## 3. How Engram Works

### 3.1 Architecture Overview

Engram is a module inserted at specific layers (typically early ones like layers 2 and 15) of a Transformer backbone. It has two phases:

1. **Retrieval**: Look up static embeddings from a hash table using the local N-gram context
2. **Fusion**: Dynamically gate and integrate the retrieved embeddings with the backbone's hidden states

### 3.2 Sparse Retrieval via Hashed N-grams

**Step 1: Tokenizer Compression**

Standard tokenizers assign different IDs to semantically equivalent tokens (e.g., "Apple" vs " apple"). Engram first collapses these into canonical IDs using NFKC normalization + lowercasing, achieving ~23% vocabulary reduction. This means N-grams that are semantically identical share the same lookup key.

**Step 2: N-gram Key Construction**

For each token position *t*, Engram constructs suffix N-grams. For example, with N=3:
- 2-gram: (token at t-1, token at t)
- 3-gram: (token at t-2, token at t-1, token at t)

**Step 3: Multi-Head Hashing**

Directly storing all possible N-gram combinations is intractable. Instead, Engram uses *K* different hash functions per N-gram order, each mapping to an embedding table of prime size *M*:

```
For each N-gram order n, for each hash head k:
    index = hash_k(n-gram) mod M
    embedding = Table[n][k][index]
```

The final memory vector is the concatenation of all retrieved embeddings across all N-gram orders and hash heads.

This is **O(1) per token** — constant-time lookup regardless of table size.

### 3.3 Context-Aware Gating

Raw lookup embeddings are static and may be noisy (hash collisions, polysemy). Engram uses the backbone's hidden state as a "query" to gate the retrieved memory:

1. Project the retrieved embedding into Key and Value vectors
2. Compute a scalar gate α ∈ (0, 1) via scaled dot-product between the hidden state (Query) and the Key, with RMSNorm for stability
3. If the retrieved memory *contradicts* the current context, α → 0 (suppressed)
4. If it *aligns*, α → 1 (fully integrated)

This is followed by a lightweight depthwise causal convolution (kernel size 4) with SiLU activation for expanded receptive field, then a residual connection back to the backbone.

### 3.4 Multi-Branch Integration

For multi-branch architectures (like DeepSeek's Manifold-Constrained Hyper-Connections with M=4 branches), Engram shares the embedding table and Value projection across all branches, but uses branch-specific Key projections. This allows each branch to independently decide how much memory to incorporate, while keeping the extra parameter cost low. The projections are fused into a single FP8 matmul for GPU efficiency.

---

## 4. Sparsity Allocation: How to Split the Parameter Budget

### 4.1 The Allocation Problem

Given a fixed total parameter budget, how should you split between MoE experts and Engram memory?

Define **allocation ratio ρ ∈ [0, 1]**:
- ρ = 1: Pure MoE (all inactive parameters are routed experts)
- ρ = 0: All inactive parameters are Engram memory
- ρ between 0 and 1: Hybrid

### 4.2 The U-Shaped Scaling Law

Experiments at two compute regimes (2×10²⁰ and 6×10²⁰ FLOPs) reveal a **U-shaped curve**:

```
Validation Loss
    ↑
    |  ×                              ×
    |    ×                          ×
    |      ×                      ×
    |        ×                  ×
    |          ×    ×  ×    ×
    |              ×      ×
    +—————————————————————————————→ ρ
    0%    20%   40%   60%   80%  100%
    (All Engram)              (All MoE)
```

Key findings:
- **Pure MoE (ρ=100%) is suboptimal** — it wastes capacity reconstructing static patterns through computation
- **Pure Engram (ρ≈0%) is also bad** — the model loses the ability to reason dynamically
- **Sweet spot: ρ ≈ 75-80%** — allocating ~20-25% of the sparse budget to Engram yields the best results
- This optimum is **stable across compute scales**

### 4.3 Infinite Memory Regime

When you decouple the memory budget (allow Engram to grow beyond the MoE parameter budget), validation loss follows a **strict power law** — linear improvement in log-space as memory slots increase. This means: more memory always helps, with diminishing but predictable returns, and zero extra compute cost.

---

## 5. Main Results

### 5.1 Pre-training Performance (262B tokens)

All models use 3.8B activated parameters. The key comparison:

| Model | Total Params | MMLU | BBH | ARC-C | HumanEval | MATH | GSM8K |
|-------|-------------|------|-----|-------|-----------|------|-------|
| Dense-4B | 4.1B | 48.6 | 42.8 | 59.3 | 26.8 | 15.2 | 35.5 |
| MoE-27B | 26.7B | 57.4 | 50.9 | 70.1 | 37.8 | 28.3 | 58.4 |
| **Engram-27B** | **26.7B** | **60.4** | **55.9** | **73.8** | **40.8** | **30.7** | **60.6** |
| Engram-40B | 39.5B | 60.6 | 57.5 | 76.4 | 38.4 | 30.6 | 62.6 |

Engram-27B uses the **exact same** total parameters and FLOPs as MoE-27B (just 17 fewer routed experts, replaced by 5.7B of Engram memory).

Surprising finding: the biggest gains aren't on knowledge tasks (where memory obviously helps), but on **reasoning** (BBH +5.0, ARC-C +3.7) and **code/math** (HumanEval +3.0, MATH +2.4). This suggests Engram doesn't just store facts — it frees up network depth for harder tasks.

### 5.2 Long-Context Performance

After long-context extension training (to 32K), Engram shows dramatic improvements in retrieval tasks:

| Metric | MoE-27B | Engram-27B |
|--------|---------|------------|
| Multi-Query NIAH | 84.2 | 97.0 |
| Frequent Words Extraction | 73.0 | 99.3 |

By delegating local dependencies to lookups, Engram frees attention capacity for global context — exactly what long-context retrieval needs.

---

## 6. Why Does Engram Help Reasoning? (Mechanistic Analysis)

### 6.1 It Effectively Increases Model Depth

Two analysis tools reveal the mechanism:

**LogitLens Analysis**: Projects each layer's hidden state through the final LM head to see "what the model would predict at this layer." Engram models reach high-confidence predictions much earlier — the early layers no longer waste time on pattern reconstruction.

**CKA (Centered Kernel Alignment)**: Compares representations between Engram and MoE models layer-by-layer. Finding: **Engram's layer 5 ≈ MoE's layer 12**. The representations at shallow Engram layers are functionally equivalent to much deeper MoE layers.

In other words, Engram effectively makes the model "deeper" without adding layers — by freeing the early layers from memorization duty.

### 6.2 Where to Place Engram?

There's a placement trade-off:
- **Too early** (layer 1): Hidden states lack context for good gating decisions
- **Too late** (deep layers): Pattern reconstruction has already consumed the early layers
- **Sweet spot**: Layer 2 is optimal for single injection. Splitting across layers 2 and 6 is even better.

One attention layer is sufficient for the gating mechanism to have useful context. After that, earlier is better for offloading static patterns.

### 6.3 Component Importance

Ablation study (most → least important):
1. **Multi-branch integration** — branch-specific gating is critical
2. **Context-aware gating** — without it, noisy retrievals hurt performance
3. **Tokenizer compression** — semantic normalization matters for hash quality
4. Short convolution — helpful but minor
5. 4-grams — slightly worse than 2+3 grams under fixed budget (dilutes capacity)

### 6.4 What Happens If You Remove Engram at Inference?

Zeroing out Engram after training reveals what it learned:
- **Factual knowledge collapses** (TriviaQA retains only 29%)
- **Reading comprehension barely affected** (C3 retains 93%)
- **Reasoning tasks partially affected** (BBH retains 67%)

This confirms: Engram is the primary knowledge store, while reasoning lives in the backbone.

---

## 7. System Efficiency: Why Engram is Practical

### 7.1 Training

Embedding tables are sharded across GPUs using standard model parallelism. All-to-All communication gathers active rows in forward pass and dispatches gradients in backward.

### 7.2 Inference (The Key Advantage)

Unlike MoE routing (which depends on runtime hidden states), Engram indices depend only on **input token IDs** — known before the forward pass even starts. This enables:

1. **Asynchronous prefetching**: While the GPU computes layer 1, the CPU prefetches Engram embeddings for layer 2 over PCIe
2. **Host memory offloading**: The entire Engram table can live in cheap CPU DRAM, not expensive GPU HBM
3. **Zipfian caching**: N-grams follow a power law distribution — a small cache in GPU HBM covers most accesses

**Benchmark**: A 100B-parameter Engram table offloaded entirely to host memory incurs only **2.8% throughput penalty** on an 8B dense backbone. With Zipfian caching, real-world overhead would be even less.

---

## 8. Relationship to Prior Work

| Approach | Key Difference from Engram |
|----------|--------------------------|
| **OverEncoding** | Averages N-gram embeddings with vocab embedding at layer 0; fails on MoE backbones |
| **SCONE** | Inference-focused, uses auxiliary f-gram model with extra training FLOPs; not iso-compute |
| **PKM / PEER / UltraMem** | Parametric key-value memories with learned routing; Engram uses deterministic hashing (cheaper, prefetchable) |
| **RETRO / RAG** | Non-parametric external retrieval; Engram is parametric and trained end-to-end |
| **Per-Layer Embeddings** | Expand capacity via massive tables but without compositional N-gram structure |

Engram's key differentiators: (1) deterministic addressing enables prefetching and offloading, (2) strict iso-parameter/iso-FLOPs evaluation protocol, (3) algorithm-system co-design.

---

## 9. Conclusion and Vision

The paper argues that **conditional memory** should be a first-class modeling primitive alongside conditional computation (MoE):

- MoE = sparse activation of neural parameters for dynamic reasoning
- Engram = sparse lookup of static embeddings for knowledge retrieval
- Together they form **two complementary axes of sparsity**

The U-shaped scaling law shows neither alone is optimal. The sweet spot allocates ~20-25% of the sparse budget to memory and ~75-80% to computation.

The vision: next-generation sparse models should have **three components**:
1. Dense backbone (attention + feed-forward) for sequential processing
2. MoE experts for conditional computation
3. Engram memory for conditional knowledge lookup

This decomposition aligns model architecture with the fundamental duality of language: dynamic reasoning + static knowledge.
