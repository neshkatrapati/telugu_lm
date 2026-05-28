---
license: apache-2.0
language:
  - te
  - en
tags:
  - telugu
  - llama
  - pretrained
  - causal-lm
  - morfessor
library_name: transformers
pipeline_tag: text-generation
---

# Pothana Base v2 — 225M Telugu Language Model

A ~225M parameter LLaMA-style decoder pretrained from scratch on a mixed Telugu (~91%) + English (~9%) corpus with a hybrid morfessor + BPE tokenizer. Designed as a strong base model for downstream retrieval-augmented and instruction fine-tuning on Telugu.

> **Status**: pretrained base model. Not yet instruction-tuned or RAG-aligned.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dvitvaai/pothana-base-v2-225M",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "dvitvaai/pothana-base-v2-225M",
    trust_remote_code=True,
)

# The tokenizer expects morfessor-segmented Telugu (see notes below).
# A morfessor model is shipped (morfessor_telugu.bin) for raw-text preprocessing.
inputs = tokenizer("మా@@కు ముఖ్యమైన ఫలితాల @@ను అంది @@స్తాయి", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0]))
```

`trust_remote_code=True` is required for:
- The model class (`PothanaForCausalLM`): LLaMA + QK-norm
- The tokenizer class (`PothanaTokenizer`): handles `@@` continuation prefix at decode

## Architecture

| | |
|---|---|
| Parameters | **222M** unique (370M on disk due to weight-sharing unroll) |
| Hidden size | 768 |
| Layers (unique) | 24 |
| Layers (effective, with weight sharing) | 48 |
| Attention heads | 16 query, 4 key/value (GQA, ratio 4:1) |
| Head dim | 48 |
| Intermediate (SwiGLU) | 2048 |
| Activation | SwiGLU |
| Norm | RMSNorm (eps=1e-6) |
| Position encoding | RoPE, θ=500,000 |
| **QK-norm** | yes (RMSNorm on Q and K, Llama 3.1 style) |
| **Tied embeddings** | no (lm_head separate from wte; +36M params for capacity) |
| Vocab size | 47,831 |
| Max context | 4,096 |

### Weight sharing (MobileLLM-LS style)

24 unique transformer blocks; each unique block runs twice in sequence (block-wise weight sharing). HF representation unrolls this to 48 layers with duplicated weights, so standard `from_pretrained()` works without custom logic.

## Tokenizer

- **Type**: `morfessor_bpe_telugu_v4` (custom)
- **Vocab**: 47,831 tokens
  - Telugu morphemes (segmented via Morfessor on the Sangraha Telugu corpus)
  - BPE subwords for non-Telugu text (8000 merges) → enables English coverage
  - Character fallback for OOV Telugu
  - 4 base special tokens: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`
  - 9 reserved retrieval special tokens (IDs 47822–47830): `<search>`, `</search>`, `<retrieved>`, `</retrieved>`, `<doc>`, `</doc>`, `<cite>`, `<think>`, `</think>`. **Unused during base pretraining** — reserved for downstream retrieval fine-tuning.
- **Continuation marker**: `@@` *prefix* on morphemes that attach to the previous word (e.g., `మా @@కు` → `మాకు`).
- **Preprocessing**: inputs are expected to be morfessor-segmented. The included `morfessor_telugu.bin` model can produce this from raw text.

### English fertility

Measured on Wikipedia samples: **~1.81 tokens/word, 0% UNK rate**. About 2× worse than a dedicated English BPE tokenizer — acceptable since English is only ~9% of training data.

## Training data

- **Telugu**: ~3.07B tokens, sourced from the [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) corpus, morfessor-segmented
- **English**: ~300M tokens (~10%) from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en), tokenized via the BPE fallback
- **Mix**: 3.37B total training tokens (91.2% Telugu / 8.8% English)
- **UNK rate** on Telugu training set: 8.8e-6 (essentially zero)

The two languages are concatenated (`train.bin` is Telugu followed by English) and the dataloader uses random uniform sampling across the full file — sequences are effectively independent draws.

## Training procedure

- **Hardware**: 1× NVIDIA B200 (192 GB HBM)
- **Wall time**: 48.6 hours
- **Total steps**: 8,000
- **Effective batch**: 512 sequences × 4,096 tokens = 2.1M tokens/step
- **Total training tokens**: ~16.8B (≈ 5 epochs over the 3.37B-token corpus)
- **Optimizer**: AdamW, β=(0.9, 0.95), weight decay 0.1, grad clip 1.0
- **Learning rate**: peak 5e-4 with **WSD schedule** (warmup 3,000 steps → stable to step 5,600 → linear decay to 5e-5 at step 8,000)
- **Loss**: cross-entropy + **z-loss** (λ=1e-4) for output normalization
- **Mixed precision**: bf16 with fp32 master weights
- **Throughput**: ~95,800 tokens/sec sustained

### Loss trajectory

| Step | val_loss (training-time) | notes |
|---:|---:|---|
| 500 | 5.5729 | start of training |
| 1,500 | 4.0242 | |
| 3,000 | 3.5619 | warmup ends |
| 5,000 | 3.3740 | end of stable-LR phase |
| 6,000 | 3.2867 | mid decay |
| 7,500 | 3.1856 | last training-time eval |
| **8,000 (final)** | **3.1631** | deterministic eval, 40 batches × 8 × 4,096 |

### Architectural tier-1 improvements over prior baselines (in this project)

| Feature | Value | Why |
|---|---|---|
| Untied embeddings | +36M params for dedicated `lm_head` | Capacity improvement, ~0.05 NLL expected |
| **QK-norm** | RMSNorm on Q, K before RoPE | Long-context stability (Llama 3.1, Cosmos) |
| z-loss | λ=1e-4 | Prevents logit drift (PaLM, Gemini) |
| 4096 context | from 2048 baseline | Headroom for downstream retrieval |
| WSD schedule | 70% stable / 30% linear decay | More efficient than cosine at this scale |
| 10% English mix | ~300M Wikipedia tokens | Cross-lingual capability for future retrieval over English sources |

## Evaluation

- **Final val loss (held-out Telugu + English mix)**: **3.1631** (perplexity ≈ 23.6)

> Comparable models in this project's history:
> - Prior engram baseline (235M, 9000 steps, 2048 ctx, no QK-norm, tied emb): val 3.42 — **0.26 NLL worse**
> - This Base v2 represents ~30% perplexity reduction over the engram baseline.

External benchmarks (IndicGLUE, TyDi-QA-Telugu, etc.) have not been run yet for this checkpoint and will be added when available.

## Intended use

This is a **pretrained base model**, not an instruction-tuned model. It is suitable as a starting point for:

- Telugu text continuation / completion experiments
- Fine-tuning for downstream tasks (classification, NER, summarization)
- **Retrieval-augmented generation (RAG) fine-tuning** — the special tokens for retrieval are already in the vocabulary; see project notes on `RETRIEVAL.md` for the planned post-training pipeline (continued pretrain → SFT → DPO → verifier)
- Research on small-scale Telugu language modeling

The model is not suitable for direct use as a chat assistant without further fine-tuning.

## Limitations

- **No instruction tuning**: zero-shot prompts will get continuation-style outputs, not Q&A-style responses.
- **Small parameter count (225M)**: limited factual knowledge; reasoning depth is modest.
- **Tokenizer requires morfessor preprocessing**: raw Telugu input must be segmented first. The included `morfessor_telugu.bin` handles this but adds a preprocessing step.
- **English fertility is suboptimal** (~1.81 tok/word vs ~0.75 for dedicated English BPE) — English-heavy use cases would benefit from a different tokenizer.
- **Telugu Wikipedia and high-quality Telugu factual data are limited** in the training corpus; the model's factual knowledge is heavily skewed toward what appears in Sangraha (general web Telugu).
- **No safety / alignment work has been done.** The base model can produce toxic, biased, or fabricated content. Do not use in production without adding appropriate guardrails.

## Citation

If you use this model, please reference:

```bibtex
@misc{pothana-base-v2-225M,
  title  = {Pothana Base v2: A 225M Telugu LLaMA-style language model with QK-norm},
  author = {Katrapati, Ganesh},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/dvitvaai/pothana-base-v2-225M}},
}
```

## Acknowledgments

- Training corpus from [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)
- English data from [Wikimedia Foundation](https://huggingface.co/datasets/wikimedia/wikipedia)
- Architecture inspired by LLaMA, MobileLLM-LS (weight sharing), Llama 3.1 (QK-norm)
- Training schedule (WSD) follows recent recommendations from the [SlowRun](https://github.com/qlabs-eng/slowrun) benchmark community

## License

Apache 2.0. Free for research and commercial use with attribution.
