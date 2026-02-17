# Engram Integration Plan for Pothana (Telugu SLM)

## Background

### What is Engram?

**Paper**: *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (DeepSeek-AI, 2025)

Current LLMs use the same expensive neural computation for two fundamentally different tasks:

1. **Compositional reasoning** — multi-step logic, inference, context understanding
2. **Knowledge retrieval** — recalling named entities, idioms, formulaic patterns

The problem: Transformers have no native lookup mechanism. They're forced to *simulate* retrieval through layers of attention and FFN. For example, to recognize "Diana, Princess of Wales", an LLM progressively composes features across 6+ layers — burning depth on what could be a simple table lookup.

**Engram** adds a **hash-based N-gram lookup table** alongside the neural network. Instead of wasting layers reconstructing static patterns through computation, the model looks them up in O(1) time via hashing, freeing network depth for actual reasoning.

### Key Results from the Paper

- **Engram-27B** (same total params and FLOPs as MoE-27B) beats the baseline across all domains:
  - Knowledge: MMLU +3.4, CMMLU +4.0
  - Reasoning: BBH +5.0, ARC-Challenge +3.7
  - Code/Math: HumanEval +3.0, MATH +2.4
  - Long-context: Multi-Query NIAH 84.2 → 97.0
- **Effective depth increase**: Engram layer 5 ≈ MoE layer 12 in representation quality
- **System efficiency**: 100B-param table offloaded to CPU incurs only 2.8% throughput penalty
- **U-shaped scaling law**: optimal allocation is ~75-80% neural compute, ~20-25% memory

---

## How Engram Works (Step by Step)

### Example

Sentence: `"this cat is yellow and that dog is red"`

**Step 1: Tokenizer Compression**

Collapse semantically equivalent tokens to canonical IDs (lowercase, NFKC normalize):

```
this=1  cat=2  is=3  yellow=4  and=5  that=6  dog=7  red=8
```

**Step 2: Build Suffix N-grams**

For each token position, look backward to form 2-grams and 3-grams:

| Position | Token  | 2-gram         | 3-gram               |
|----------|--------|----------------|-----------------------|
| 0        | this   | (∅, this)      | (∅, ∅, this)          |
| 1        | cat    | (this, cat)    | (∅, this, cat)        |
| 2        | is     | (cat, is)      | (this, cat, is)       |
| 3        | yellow | (is, yellow)   | (cat, is, yellow)     |
| 4        | and    | (yellow, and)  | (is, yellow, and)     |
| 5        | that   | (and, that)    | (yellow, and, that)   |
| 6        | dog    | (that, dog)    | (and, that, dog)      |
| 7        | is     | (dog, is)      | (that, dog, is)       |
| 8        | red    | (is, red)      | (dog, is, red)        |

**Step 3: Multi-Head Hashing → Embedding Lookup**

Each N-gram is hashed via K=8 independent hash functions into embedding tables of prime size M:

```
For position 7 ("is"), hash head 1:
  2-gram: hash_1("dog", "is") = 4831207  →  Table[2gram][head1][4831207] → 32-dim vector
  3-gram: hash_1("that", "dog", "is") = 12044891  →  Table[3gram][head1][12044891] → 32-dim vector
```

Repeat for all 8 heads × 2 orders = 16 lookups. Concatenate:

```
e_t = [2gram_h1 | ... | 2gram_h8 | 3gram_h1 | ... | 3gram_h8]
    = 16 × 32-dim = 512-dim memory vector
```

**This is O(1)** — just array indexing, no matrix multiplications.

**Step 4: Context-Aware Gating**

The retrieved embedding `e_t` is static and may be noisy (hash collisions). Use the backbone's hidden state `h_t` to gate it:

```
k_t = W_K @ e_t                                    # project to key
v_t = W_V @ e_t                                    # project to value
α_t = sigmoid(RMSNorm(h_t)^T · RMSNorm(k_t) / √d) # scalar gate ∈ (0,1)
output_t = α_t · v_t                                # gated output
```

If `e_t` aligns with the current context → α≈1 (use memory). If hash collision or irrelevant → α≈0 (suppress).

**Step 5: Conv + Residual**

```
Y = SiLU(Conv1D(RMSNorm(gated_output))) + gated_output   # lightweight conv
h_t = h_t + Y                                             # residual connection
```

Then the normal Attention → MLP layers continue.

---

## Why Engram is Relevant for Pothana

Our Telugu model uses a Morfessor tokenizer that segments words into morphemes with `@@` markers:

```
విద్యార్థులుకు → విద్యార్థు@@ లు@@ కు
అందమైనది    → అందమైన@@ ది
ప్రభుత్వం     → ప్రభుత్వం (single token, no split)
```

These morpheme N-grams are **exactly** the kind of local, stereotyped patterns Engram is designed to memorize:

- `విద్యార్థు@@ లు@@` — extremely common Telugu morpheme pattern (student + plural)
- `అందమైన@@ ది` — adjective + suffix pattern
- `లో@@ ని` — postposition patterns

Currently, our model burns early transformer layers learning that `విద్యార్థు@@ లు@@ కు` = "విద్యార్థులుకు" (students + dative case). Engram can hand that mapping to the model for free via lookup, freeing those layers for actual reasoning about the sentence content.

---

## Our Current Architecture

```
GPTConfig:
  block_size:  2048    (context length)
  vocab_size:  ~42K    (Morfessor + BPE + char fallback)
  n_layer:     20      (transformer blocks)
  n_head:      16      (attention heads)
  n_embd:      1024    (hidden dimension)
  dropout:     0.1
  bias:        False
  rope_theta:  10000.0

Derived:
  head_dim:           64   (1024 / 16)
  intermediate_size:  2816 (SwiGLU: round(2 * 1024 * 4/3, 256))
  total_params:       ~300M
```

### Current Block Structure (Pre-norm, Single-stream)

```python
class Block:
    ln_1:  RMSNorm(1024)
    attn:  CausalSelfAttention(config)
    ln_2:  RMSNorm(1024)
    mlp:   SwiGLUMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x
```

### Current GPT Forward Pass

```python
def forward(self, idx, targets=None):
    tok_emb = self.transformer.wte(idx)          # (B, T) → (B, T, 1024)
    x = self.transformer.drop(tok_emb)
    freqs_cis = self.freqs_cis[:T]

    for block in self.transformer.h:             # 20 blocks
        x = block(x, freqs_cis)

    x = self.transformer.ln_f(x)                # final RMSNorm
    logits = self.lm_head(x)                    # (B, T, vocab_size)
    loss = cross_entropy(logits, targets)
    return logits, loss
```

### Current Training Loop

```
Optimizer:    AdamW (lr=3e-4, betas=(0.9, 0.95))
Weight decay: 0.1 for matrices, 0.0 for norms/biases
Batch:        32 micro × 4 accumulation = 128 sequences = 262K tokens/step
Schedule:     Cosine decay (3e-4 → 3e-5) with 500-step warmup
Precision:    bf16 mixed precision with GradScaler
Compilation:  torch.compile enabled
Grad clip:    1.0
```

---

## Implementation Plan

### New Config Fields

```python
@dataclass
class GPTConfig:
    # ... existing fields ...

    # Engram config
    engram_layers: tuple = (1, 9)       # which block indices get Engram
    engram_n_gram: int = 3              # max N-gram order (uses 2-grams and 3-grams)
    engram_n_heads: int = 8             # hash heads per N-gram order
    engram_table_size: int = 500009     # prime number, entries per head per order
    engram_dim: int = 512               # total memory vector dimension
                                        # = n_gram_orders × n_heads × (engram_dim / orders / heads)
                                        # = 2 × 8 × 32 = 512
```

### New Module: `EngramModule`

```
EngramModule(config)
├── compress_map: IntTensor[vocab_size]     # token_id → canonical_id (precomputed buffer)
│
├── tables: ParameterList                    # embedding tables
│   ├── order=2: K=8 tables, each (M, d_per_head)   # M=500009, d=32
│   └── order=3: K=8 tables, each (M, d_per_head)
│
├── hash_weights: IntTensor[n_orders, K]     # random hash multipliers (buffer)
│
├── ln_mem: RMSNorm(engram_dim)              # normalize retrieved memory
├── W_K: Linear(engram_dim, n_embd)          # key projection
├── W_V: Linear(engram_dim, n_embd)          # value projection
├── ln_q: RMSNorm(n_embd)                   # query norm (for hidden state)
├── ln_k: RMSNorm(n_embd)                   # key norm
│
├── conv: Conv1d(n_embd, n_embd,             # depthwise causal conv
│         kernel_size=4, dilation=3,
│         groups=n_embd, padding='causal')
├── ln_conv: RMSNorm(n_embd)
└── act: SiLU()
```

**Forward pass:**

```python
def forward(self, h, input_ids):
    """
    h:         (B, T, d_model)   — current hidden state (after ≥1 attention layer)
    input_ids: (B, T)            — original token IDs
    returns:   (B, T, d_model)   — residual addition to h
    """
    # 1. Compress token IDs
    cids = self.compress_map[input_ids]                    # (B, T)

    # 2. Build N-gram keys and hash → retrieve embeddings
    parts = []
    for n in [2, 3]:                                       # N-gram orders
        for k in range(self.n_heads):                      # hash heads
            ngram_key = combine_ngram(cids, n, t)          # suffix N-gram
            idx = hash_fn(ngram_key, self.hash_weights[n, k]) % self.table_size
            emb = self.tables[n][k][idx]                   # (B, T, d_per_head)
            parts.append(emb)
    e = torch.cat(parts, dim=-1)                           # (B, T, engram_dim)

    # 3. Context-aware gating
    k = self.W_K(e)                                        # (B, T, d_model)
    v = self.W_V(e)                                        # (B, T, d_model)
    q = self.ln_q(h)
    k = self.ln_k(k)
    alpha = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / math.sqrt(d_model))
    gated_v = alpha * v                                    # (B, T, d_model)

    # 4. Causal conv + residual
    y = self.ln_conv(gated_v)
    y = self.act(self.conv(y.transpose(1, 2)).transpose(1, 2)) + gated_v

    return y                                               # add to h via residual
```

### Modified `TransformerBlock`

```python
class Block:
    ln_1:   RMSNorm(1024)
    attn:   CausalSelfAttention(config)
    ln_2:   RMSNorm(1024)
    mlp:    SwiGLUMLP(config)
    engram: EngramModule(config) or None     # ← NEW (only at configured layers)

    def forward(self, x, freqs_cis, input_ids=None):
        if self.engram is not None and input_ids is not None:
            x = x + self.engram(x, input_ids)          # ← NEW: memory lookup
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x
```

### Modified `GPT.forward()`

```python
def forward(self, idx, targets=None):
    tok_emb = self.transformer.wte(idx)
    x = self.transformer.drop(tok_emb)
    freqs_cis = self.freqs_cis[:T]

    for block in self.transformer.h:
        x = block(x, freqs_cis, input_ids=idx)     # ← pass input_ids through

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    ...
```

### Tokenizer Compression Map

Build from our `tokenizer.json` at model init:

```python
def build_compress_map(token_to_id):
    """Map raw token IDs → canonical IDs.

    Collapse:
    - "విద్యార్థు@@" and "విద్యార్థు" → same canonical ID
    - "Apple" and "apple" → same canonical ID
    - NFKC normalization
    """
    canonical = {}
    next_id = 0
    compress = torch.zeros(len(token_to_id), dtype=torch.long)

    for token, tid in token_to_id.items():
        # Normalize: strip @@, lowercase, NFKC
        norm = unicodedata.normalize('NFKC', token.replace('@@', '').lower().strip())
        if norm not in canonical:
            canonical[norm] = next_id
            next_id += 1
        compress[tid] = canonical[norm]

    return compress   # register_buffer in EngramModule
```

This achieves similar compression to the paper's ~23% vocab reduction.

### Hash Function

Lightweight multiplicative-XOR hash (from the paper):

```python
def hash_ngram(canonical_ids, weights, table_size):
    """
    canonical_ids: (B, T, n) — the n tokens forming the N-gram
    weights:       (n,)      — random hash multipliers per position
    table_size:    int       — prime number M
    returns:       (B, T)    — hash indices
    """
    h = torch.zeros(B, T, dtype=torch.long, device=device)
    for i in range(n):
        h = h ^ (canonical_ids[:, :, i] * weights[i])
    return h % table_size
```

### Optimizer Group Changes

```python
# Existing groups
decay_params    = [p for n, p in params if p.dim() >= 2 and 'engram.tables' not in n]
nodecay_params  = [p for n, p in params if p.dim() < 2  and 'engram.tables' not in n]

# NEW: Engram embeddings get higher LR, no weight decay
engram_emb_params = [p for n, p in params if 'engram.tables' in n]
engram_other      = [p for n, p in params if 'engram' in n and 'tables' not in n and p.dim() >= 2]

optim_groups = [
    {"params": decay_params,      "weight_decay": 0.1,  "lr": base_lr},
    {"params": nodecay_params,    "weight_decay": 0.0,  "lr": base_lr},
    {"params": engram_emb_params, "weight_decay": 0.0,  "lr": base_lr * 5},   # 5× LR for embeddings
    {"params": engram_other,      "weight_decay": 0.1,  "lr": base_lr},       # projections: normal
]
```

### Initialization

- **Embedding tables**: Normal(0, 0.02) — same as vocab embeddings
- **W_K, W_V projections**: Normal(0, 0.02)
- **Conv weights**: **Zeros** — critical! This makes the Engram output zero at init, so the model starts as if Engram doesn't exist. It gradually learns to use memory.
- **RMSNorm**: Ones (default)

---

## Parameter Budget

For our 300M-param dense model, Engram adds memory without computation:

```
Per hash head per N-gram order:
  table_size × d_per_head = 500,009 × 32 = 16M params

Total embedding tables:
  2 orders × 8 heads × 16M = 256M params (in float32)
  = 256M × 4 bytes = 1.0 GB in float32
  = 256M × 2 bytes = 512 MB in bf16

Projection + conv + norms (per Engram module):
  W_K: 512 × 1024 = 524K
  W_V: 512 × 1024 = 524K
  Conv1d: 1024 × 4 = 4K (depthwise)
  Norms: ~5K
  ≈ 1M params per module × 2 modules = 2M

Total Engram params: ~258M
New total model: ~300M (backbone) + ~258M (Engram) ≈ 558M total params
Activated params per token: still ~300M (Engram is O(1) lookups, not FLOPs)
```

**Alternative smaller config** if memory is tight:

```
table_size = 200003, d_per_head = 16
→ 2 × 8 × 200,003 × 16 = 51M params = 200 MB in float32
→ New total: ~350M params
```

---

## Layer Placement Strategy

The paper found:
- **Layer 2 is optimal** for single injection (one attention layer for context, then inject early)
- **Splitting across layers 2 and 6** is even better (early + mid intervention)

For our 20-layer model, we use **layers 1 and 9** (0-indexed):
- **Layer 1**: After one attention pass, inject memory to offload local pattern reconstruction
- **Layer 9**: Mid-network injection for deeper contextual gating

This also provides the system advantage of compute overlap for prefetching (Section 2.5 of the paper).

---

## Training Strategy

### Phase 1: Continue Pre-training with Engram (Recommended)

1. Load the existing 300M checkpoint (`best.pt`)
2. Add Engram modules (conv initialized to zero → model starts identical to baseline)
3. Continue training on the same data with identical hyperparameters
4. Engram gradually learns to store morpheme patterns

```bash
python train_gpt.py \
    --checkpoint ./checkpoints/best.pt \
    --engram-layers 1,9 \
    --engram-table-size 500009 \
    --engram-dim 512
```

Load with `strict=False` to handle missing Engram keys in old checkpoint.

### Phase 2: Scale Memory (If Phase 1 Succeeds)

The paper shows log-linear improvement with table size. Once Phase 1 validates the approach:

- Increase `table_size` from 500K → 2M → 10M
- Increase `engram_dim` from 512 → 1024
- Add higher-order N-grams (4-grams) if budget allows

### Monitoring

Track during training:
- **Gate activation statistics**: `mean(α_t)` per layer — should be >0 and not saturated
- **Gate selectivity**: `std(α_t)` across positions — should show high variance (selective activation)
- **Validation loss**: should improve over baseline within first 1-2K steps

---

## What We Do NOT Need (Simplifications vs Paper)

| Paper Feature | Our Approach | Why |
|--------------|-------------|-----|
| MoE backbone + allocation ratio | Dense backbone, just add memory | We're dense, not MoE — no allocation trade-off |
| Multi-branch (mHC, M=4) integration | Single-stream residual | We use standard pre-norm Transformer |
| Branch-specific Key projections | Single W_K | No branches to differentiate |
| All-to-All sharding | Single GPU | Tables fit in one GPU's memory |
| Host memory offloading + prefetch | Keep on GPU | ~512MB-1GB fits easily in GPU HBM |
| 30-layer backbone | 20-layer backbone | Our scale is smaller |

---

## Files to Modify

| File | Changes |
|------|---------|
| `train_gpt.py` | Add `EngramModule` class (~150 lines). Add engram fields to `GPTConfig`. Modify `Block.__init__` and `Block.forward`. Modify `GPT.__init__` and `GPT.forward` to pass `input_ids`. Modify optimizer grouping for Engram params. |
| `train_gpt.py` | Add `build_compress_map()` utility. Modify checkpoint save/load for Engram state. |
| `convert_to_hf.py` | Add Engram weight mapping to HF format (new state_dict keys). |
| `inference.py` | No changes needed — `input_ids` already available in the pipeline. |

---

## Expected Impact

Based on the paper's results scaled to our regime:

1. **Knowledge tasks**: Significant improvement — Engram memorizes Telugu morpheme patterns, named entities, common phrases
2. **Reasoning**: Moderate improvement — freed early layers can do more reasoning
3. **Long-context**: Improvement expected — attention freed from local dependencies
4. **Training efficiency**: Same FLOPs but more effective use of parameters
5. **Inference cost**: Negligible overhead — O(1) lookups add ~0 FLOPs

The Morfessor `@@` tokenization is a particularly good fit because it creates *exactly* the kind of local, stereotyped N-gram patterns that Engram excels at memorizing.
