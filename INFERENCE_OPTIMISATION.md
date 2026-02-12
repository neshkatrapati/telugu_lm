# Telugu LLaMA — Inference Optimisation Plan

## Current Baseline

- **Model**: 300M params, LLaMA-style (RoPE + SwiGLU + RMSNorm)
- **Inference**: Raw PyTorch, bf16/fp32, no caching
- **CPU speed**: ~5-10 tokens/sec
- **Memory**: ~1.2GB

---

## Phase 1: PyTorch-level Optimisations

Target: **60-120 tokens/sec on CPU**, ~400MB memory

### Step 1 — KV Cache (3-4x speedup)

**Problem**: Current `generate()` recomputes attention over the entire sequence at every step. At token 200, it reprocesses all 200 tokens to produce 1 new token. This is O(n^2) total work for n generated tokens.

**Solution**: Cache the Key and Value tensors from each attention layer. On each new token, only compute Q/K/V for the new token position, concatenate K/V with the cache, and compute attention against the full cached K/V.

**Changes required**:
- Add `KVCache` class that stores `(batch, n_heads, seq_len, head_dim)` tensors per layer
- Modify `CausalSelfAttention.forward()` to accept and return `(k_cache, v_cache)`
  - On first pass (prefill): process full prompt, populate cache
  - On subsequent passes (decode): process single token, append to cache
- Modify `Block.forward()` to pass cache through
- Modify `GPT.forward()` to accept `past_kv` list and return updated cache
- Modify `GPT.generate()` (or write new `generate_with_cache()`) to use the cache
- RoPE positions must use absolute position (not relative to input length) so cached K rotations remain correct

**Expected result**:
- Speed: 5-10 → 20-40 tokens/sec
- Memory: +200MB for cache (negligible at 2048 context)
- Quality: Identical — mathematically equivalent output

### Step 2 — INT8 Dynamic Quantization (2x speedup)

**Problem**: fp32 weights are 4 bytes each. CPU memory bandwidth is the bottleneck for inference — every token generation reads the entire model from RAM.

**Solution**: Quantize Linear layer weights from fp32 to int8 (1 byte). Use `torch.ao.quantization.quantize_dynamic` which keeps activations in fp32 but stores weights as int8, dequantizing on the fly during matmul.

**Changes required**:
- After loading the model checkpoint, apply dynamic quantization:
  ```python
  import torch.ao.quantization as quant
  model = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
  ```
- That's it — one function call. Applies to all `nn.Linear` layers (attention projections, MLP layers, lm_head)
- RMSNorm layers are unaffected (no Linear weights)

**Expected result**:
- Speed: 20-40 → 40-80 tokens/sec
- Memory: ~1.2GB → ~400MB (weights 4x smaller)
- Quality: ~99% same — negligible perplexity increase (<0.1)

### Step 3 — torch.compile CPU Backend (1.5x speedup)

**Problem**: Python overhead between operations, and unfused elementwise ops (RMSNorm multiply, SwiGLU gate * up, softmax) leave performance on the table.

**Solution**: Use `torch.compile()` which traces the model, fuses operations, and generates optimised CPU kernels via TorchInductor.

**Changes required**:
- After loading and quantizing the model:
  ```python
  model = torch.compile(model, mode="reduce-overhead")
  ```
- First generation will be slow (compilation overhead ~30-60 seconds)
- Subsequent generations run at full speed
- Note: torch.compile + quantized models requires PyTorch >= 2.2

**Expected result**:
- Speed: 40-80 → 60-120 tokens/sec
- Memory: No significant change
- Quality: Identical — compile does not change numerics

### Phase 1 Summary

| Step | Optimisation | Speed (tok/s) | Memory | Quality |
|------|-------------|---------------|--------|---------|
| Baseline | None | 5-10 | 1.2GB | 100% |
| 1 | KV Cache | 20-40 | 1.4GB | Identical |
| 1+2 | + INT8 Quantization | 40-80 | 400MB | ~99% |
| 1+2+3 | + torch.compile | 60-120 | 400MB | ~99% |

---

## Phase 2: GGUF Export + llama.cpp

Target: **150-300 tokens/sec on CPU**, ~150MB memory

### Overview

llama.cpp is a C/C++ inference engine with hand-optimised kernels for transformer models. It uses the GGUF file format and supports aggressive quantization (Q4_0, Q4_K_M, Q8_0). It includes built-in KV caching, SIMD vectorisation (AVX2/AVX-512/ARM NEON), and multi-threaded inference.

This replaces Phase 1 entirely — GGUF inference does not use PyTorch at all.

### Step 1 — Write GGUF Exporter

**Problem**: Our model architecture is a custom LLaMA-style implementation. llama.cpp expects weights in GGUF format with specific tensor names matching its internal architecture definitions.

**Solution**: Write a Python script (`export_gguf.py`) that:
1. Loads our checkpoint (`.pt` file)
2. Maps our tensor names to llama.cpp's expected names:
   ```
   Our name                          → GGUF name
   transformer.wte.weight            → token_embd.weight
   transformer.h.{i}.ln_1.weight    → blk.{i}.attn_norm.weight
   transformer.h.{i}.attn.c_q.weight → blk.{i}.attn_q.weight
   transformer.h.{i}.attn.c_k.weight → blk.{i}.attn_k.weight
   transformer.h.{i}.attn.c_v.weight → blk.{i}.attn_v.weight
   transformer.h.{i}.attn.c_proj.weight → blk.{i}.attn_output.weight
   transformer.h.{i}.ln_2.weight    → blk.{i}.ffn_norm.weight
   transformer.h.{i}.mlp.w_gate.weight → blk.{i}.ffn_gate.weight
   transformer.h.{i}.mlp.w_up.weight   → blk.{i}.ffn_up.weight
   transformer.h.{i}.mlp.w_down.weight → blk.{i}.ffn_down.weight
   transformer.ln_f.weight           → output_norm.weight
   lm_head.weight                    → output.weight
   ```
3. Writes model hyperparameters (n_layer, n_head, n_embd, vocab_size, context_length, rope_theta) as GGUF metadata
4. Writes weights in fp16 format to the GGUF file
5. Exports tokenizer vocabulary as GGUF token metadata

**Dependencies**: `gguf` Python package (`pip install gguf`)

**Estimated code**: ~200 lines

### Step 2 — Quantize GGUF

**Solution**: Use llama.cpp's built-in `llama-quantize` tool to quantize the fp16 GGUF to various levels:

```bash
# Q8_0 — 8-bit, best quality, ~300MB
llama-quantize telugu-llama-f16.gguf telugu-llama-q8.gguf Q8_0

# Q4_K_M — 4-bit mixed, good balance, ~150MB
llama-quantize telugu-llama-f16.gguf telugu-llama-q4km.gguf Q4_K_M

# Q4_0 — 4-bit basic, smallest, ~150MB
llama-quantize telugu-llama-f16.gguf telugu-llama-q4.gguf Q4_0
```

**Quantization options for 300M model**:

| Format | Size | Quality | Speed |
|--------|------|---------|-------|
| F16 | ~600MB | 100% | 80-120 tok/s |
| Q8_0 | ~300MB | ~99.5% | 120-200 tok/s |
| Q4_K_M | ~150MB | ~97% | 150-300 tok/s |
| Q4_0 | ~150MB | ~96% | 150-300 tok/s |

### Step 3 — Run with llama.cpp

```bash
# Interactive chat
llama-cli -m telugu-llama-q4km.gguf -p "తెలుగు భాష" -n 200 --temp 0.8 --top-k 50

# Server mode (REST API)
llama-server -m telugu-llama-q4km.gguf --port 8080
# Then: curl http://localhost:8080/completion -d '{"prompt": "తెలుగు భాష", "n_predict": 200}'
```

**Threading**: llama.cpp auto-detects CPU cores. For max throughput:
```bash
llama-cli -m telugu-llama-q4km.gguf -t 8 -p "తెలుగు భాష"
```

### Phase 2 Summary

| Step | Action | Output |
|------|--------|--------|
| 1 | Write `export_gguf.py` | `telugu-llama-f16.gguf` |
| 2 | Quantize with llama-quantize | `telugu-llama-q4km.gguf` (~150MB) |
| 3 | Run with llama-cli or llama-server | 150-300 tok/s on CPU |

### Deployment Footprint (Phase 2)

- **Model file**: ~150MB (Q4_K_M)
- **Runtime**: llama.cpp binary (~2MB, statically linked, no Python needed)
- **RAM at runtime**: ~300MB
- **Dependencies**: None (standalone binary)
- **Platforms**: Linux, macOS (Intel + Apple Silicon), Windows

---

## Comparison: Phase 1 vs Phase 2

| | Phase 1 (PyTorch) | Phase 2 (GGUF) |
|---|---|---|
| Speed | 60-120 tok/s | 150-300 tok/s |
| Memory | ~400MB | ~300MB |
| Model size on disk | ~1.2GB (pt) + ~400MB (quantized in RAM) | ~150MB (Q4_K_M) |
| Dependencies | Python, PyTorch, morfessor | llama.cpp binary only |
| Ease of implementation | Moderate (KV cache rewrite) | Moderate (GGUF exporter) |
| Flexibility | Full Python control | C++ binary, less customisable |
| Deployment | Needs Python env | Single binary + model file |
| Recommended for | Development, experimentation | Production, edge deployment |
