# Engram Implementation Plan (Stashed)

Stashed for later implementation. Complete the architecture improvements and tokenizer first.

## Overview

Engram = n-gram embedding lookup module for transformer, inspired by DeepSeek paper.
Adds pre-computed n-gram information to token representations at each layer.

## Architecture Design

### N-gram Vocabulary
- **Source**: KenLM ARPA file (trigram model trained on tokenized corpus)
- **Filtering**: Entropy-based scoring: `score = P(joint) × surprisal`
  - Bigrams: `score = 10^(log_P_a + log_P_b|a) × (-log_P_b|a)`
  - Trigrams: uses bigram probability lookup with backoff fallback
- **Target sizes**: ~200K bigrams, ~100K trigrams (configurable via `--max-bigrams`, `--max-trigrams`)
- **Script**: `filter_ngrams_entropy.py` — reads ARPA, scores, selects top-K, outputs `ngram_vocab.json`
- **Output format**: `ngram_vocab.json` with version=3, method="entropy_mdl"

### BLT Paper Insights (Dynamic Patching)
- BLT uses entropy-based segmentation — patch boundaries where next-byte entropy is high
- Maps to n-gram selection: high-entropy n-grams = hard to predict = worth memorizing
- Low-entropy n-grams = predictable = transformer handles them fine
- This is exactly what our `count × surprisal` scoring captures

### MDL (Minimum Description Length) Framing
- Each n-gram has a cost: embedding parameters (embd_dim floats)
- Each n-gram has a benefit: information it explains in the corpus
- Keep n-gram if benefit > cost
- Implemented via the entropy-based top-K selection

## Model Integration

### EngramModule Design
```
Per transformer layer:
  1. Look up n-gram embeddings for all bigram/trigram windows in input
  2. Project to layer dimension via per-layer linear
  3. Gate with learned scalar (initialized near 0)
  4. Optional 1D conv for smoothing
  5. Add to residual stream
```

### Embedding Tables (shared across layers)
- Bigram table: `nn.Embedding(n_bigrams, engram_dim)` — e.g., 200K × 128
- Trigram table: `nn.Embedding(n_trigrams, engram_dim)` — e.g., 100K × 128
- Unigram safety net: reuse the main `wte` embedding (no extra params)
- Tables shared between ALL layers (only projections are per-layer)

### Per-Layer Components
- `proj`: `nn.Linear(engram_dim, n_embd)` — project engram to model dim
- `gate`: `nn.Parameter(torch.zeros(1))` — learned gating scalar, sigmoid applied
- `conv`: `nn.Conv1d(n_embd, n_embd, kernel_size=3, padding=1)` — optional smoothing

### GPTConfig Additions (when implementing)
```python
# Engram config
use_engrams: bool = False
engram_vocab_path: str = ""     # path to ngram_vocab.json
engram_dim: int = 128           # n-gram embedding dimension
engram_conv: bool = True        # use conv smoothing
```

### Block Modification
```python
# In Block.forward():
def forward(self, x, freqs_cis, engram_emb=None):
    x = x + self.attn(self.ln_1(x), freqs_cis)
    x = x + self.mlp(self.ln_2(x))
    if engram_emb is not None:
        x = x + self.engram_gate.sigmoid() * self.engram_proj(engram_emb)
    return x
```

### GPT.forward() Changes
```python
# Before the block loop:
if self.engram_module is not None:
    engram_emb = self.engram_module(idx)  # (B, T, engram_dim)
else:
    engram_emb = None

# In the block loop:
for block in self._block_schedule:
    x = block(x, freqs_cis, engram_emb)
```

### N-gram Index Lookup (in EngramModule.forward)
```python
# For each position t, look up:
#   bigram_id = bigram_vocab.get((token[t-1], token[t]))
#   trigram_id = trigram_vocab.get((token[t-2], token[t-1], token[t]))
# Sum the embeddings (with masking for OOV)
```

## Optimizer Grouping
- Engram embedding tables: weight_decay=0.0 (like regular embeddings)
- Engram projections: weight_decay=0.1 (like other linear layers)
- Engram gates: weight_decay=0.0 (1D params)
- Engram conv: weight_decay=0.1

## Training Monitoring
- Log gate values per layer to W&B (how much each layer uses engram info)
- Log fraction of positions with valid bigram/trigram lookups

## Checkpoint Handling
- Save engram_vocab_path in config
- Engram tables in state_dict like any other module
- When resuming: verify ngram_vocab.json matches

## Parameter Budget
- 200K bigrams × 128 dim = 25.6M params
- 100K trigrams × 128 dim = 12.8M params
- Per-layer projections (24 layers): 24 × 128 × 1024 = 3.15M
- Per-layer gates: 24 × 1 = negligible
- Total engram overhead: ~41.5M params (~13% of base model)

## Dependencies
- Requires: trained tokenizer (48K SentencePiece), KenLM ARPA file, filter_ngrams_entropy.py output
- Must be implemented AFTER architecture improvements (GQA, depth, etc.)

## Files
- `filter_ngrams_entropy.py` — n-gram filtering (DONE)
- `bin_to_text.py` — convert train.bin to text for KenLM (DONE)
- `build_ngram_vocab.py` — custom n-gram counting (DEPRECATED, use KenLM)
- `train_gpt.py` — model integration (TODO)
