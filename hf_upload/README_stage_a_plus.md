---
license: apache-2.0
language:
  - te
  - en
tags:
  - telugu
  - tenglish
  - code-mixing
  - llama
  - pretrained
  - causal-lm
  - morfessor
library_name: transformers
pipeline_tag: text-generation
base_model: dvitvaai/pothana-base-v2-225M
---

# Pothana Stage A+ — 230M Telugu LM with Roman Telugu (Tenglish) Capability

Stage A+ extends [`dvitvaai/pothana-base-v2-225M`](https://huggingface.co/dvitvaai/pothana-base-v2-225M) with **code-mix and Roman Telugu (Tenglish) capabilities**. The model can now read and write Telugu in three styles:

1. **Pure Telugu script** — same as Base v2
2. **Code-mixed (Telugu + English script)** — e.g., "నేను meeting కి వెళ్తున్నాను"
3. **Roman Telugu (Tenglish)** — e.g., "naku rendu cinemalu chudaalani undi"

Designed for mobile deployment where Indian users mix scripts freely.

> **Status**: pretrained base model with code-mix capability. Not yet instruction-tuned. Intended as a starting point for retrieval-augmented or instruction fine-tuning.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dvitvaai/pothana-stage-a-plus-225M",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "dvitvaai/pothana-stage-a-plus-225M",
    trust_remote_code=True,
)

# Telugu input
inputs = tokenizer("నేను రేపు ఆఫీసుకు వెళ్లాలి", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0]))

# Roman Telugu input (no morfessor preprocessing needed)
inputs = tokenizer("naku rendu cinemalu chudaalani undi", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80, temperature=0.7, top_p=0.9)
print(tokenizer.decode(out[0]))
```

`trust_remote_code=True` is required for the custom `PothanaForCausalLM` (Llama + QK-norm) and the `PothanaTokenizer` (handles `@@` continuation prefix).

## What's new vs Base v2

| | Base v2 | Stage A+ |
|---|---|---|
| Vocab size | 47,831 | **52,831** (+5,000 Roman Telugu word tokens) |
| Parameters | 222M | **230M** (+8M from new embedding rows) |
| Telugu capability | ✓ | ✓ |
| Code-mix (Te+En script) | weak | **strong** |
| Roman Telugu reading | weak | **strong** |
| Roman Telugu writing | weak | moderate |
| Retrieval-format `<retrieved>` recognition | ✓ | ✓ |

## Training pipeline summary

```
Base v2 (val=3.16, 49h)
   ↓ Stage A: retrieval-aware continued pretrain (val=3.05, 6h)
   ↓ Resize vocab 47,831 → 52,831 (add top-5K Roman word tokens, smart-init)
   ↓ Stage A+: code-mix continued pretrain (200 steps, ~3.4 epochs on 31M-token codemix corpus)
   ↓ [THIS MODEL]
```

### Stage A+ specifics

- **Data**: 12,258 Telugu chunks rewritten by Gemini 2.0 Flash into three formats:
  - `codemix_te_en`: natural code-mixing (Te-script + En-script)
  - `codemix_roman`: same code-mixing, all-Roman (phone-typed Tenglish)
  - `telugu_roman`: pure Telugu in Roman script
  - Plus 10% original Telugu (anti-forgetting buffer)
- **Total tokens**: 31.2M (focused continued pretrain — small but enough)
- **Tokenizer extension**: top-5K most-frequent Roman Telugu words promoted from BPE-fallback to direct tokens (~33% compression on Roman content)
- **Training**: B200, effective batch 128 × seq 4096, LR 2e-5, WSD schedule, ~20 min wall time

## Tokenizer

`morfessor_bpe_telugu_v4-v6`:
- 47,831 v4 tokens (morfessor Telugu morphemes + BPE merges for non-Telugu)
- **+5,000 Roman Telugu word tokens** (e.g., `nunchi`, `prabhutvam`, `kosam`, `mukhyamantri`)
- 9 retrieval special tokens (IDs 47822–47830, unused for now)
- Total: **52,831 tokens**

The top-5K Roman tokens give massive compression: a Roman word like `prabhutvam` (was 5 BPE subwords) → now 1 token.

### Tokenizer fertility

- Telugu (segmented): same as v4
- English: 1.81 tokens/word (unchanged)
- Roman Telugu (Tenglish): **~2.5 tokens/word on common forms** (vs ~5 with pure BPE before)

## Architecture

| | |
|---|---|
| Parameters | **230M** unique (378M on disk due to weight-sharing unroll) |
| Hidden size | 768 |
| Layers (unique) | 24 |
| Layers (effective with weight sharing) | 48 |
| Attention | GQA 16Q / 4KV, head_dim 48 |
| MLP | SwiGLU, intermediate 2048 |
| Norm | RMSNorm (eps=1e-6) |
| Position | RoPE, θ=500,000 |
| **QK-norm** | yes |
| **Tied embeddings** | no |
| Vocab | 52,831 |
| Max context | 4,096 |

## What this model is good at

1. **Reading code-mixed Telugu** — handles "నేను meeting కి వెళ్తున్నాను" naturally
2. **Reading Roman Telugu** — handles "naku meeting undi" via direct tokens for common words
3. **Generating coherent Telugu prose** — short-to-medium length news/literature-style output
4. **Generating natural code-mixed Telugu** — mixes English nouns into Telugu sentences

## Limitations

- **Loops at low temperature** — like most 225M base models, gets stuck in repetition with greedy / low-temp sampling. Use temp=0.7+ and `repetition_penalty=1.15` for cleaner output.
- **Roman Telugu generation is weaker than reading** — model produces fragmented Roman output even though it reads cleanly. Will improve with Stage B SFT (planned).
- **Retrieval grounding is NOT yet trained** — model accepts `<retrieved>...</retrieved>` format from Stage A, but doesn't yet condition answers on retrieved content. This is intentional: grounded retrieval is taught at Stage B (SFT on synthetic traces).
- **No instruction tuning** — base model only. Zero-shot prompts get continuation-style outputs, not Q&A behavior.
- **Factual coverage limited** to Sangraha corpus (general Telugu web/news) + 8.8% English Wikipedia from Base v2.

## Intended use

Starting point for downstream work:

1. **Retrieval-augmented fine-tuning** — the natural next step (Stage B)
2. **Telugu / Tenglish instruction tuning** — possible with appropriate dataset
3. **Telugu text classification, NER, summarization** — fine-tune with task data
4. **Research on small-scale Telugu language modeling**

## Evaluation

- **Stage A val_loss**: 3.05 (on retrieval-mixed corpus)
- **Stage A+ best_val_loss**: ~3.0 (codemix corpus, 3.4 epochs)

External benchmarks (IndicGLUE, TyDi-QA-Telugu) have not been run yet.

## Citation

```bibtex
@misc{pothana-stage-a-plus-225M,
  title  = {Pothana Stage A+: A 230M Telugu LM with Roman Telugu and code-mix capability},
  author = {Katrapati, Ganesh},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/dvitvaai/pothana-stage-a-plus-225M}},
}
```

## Acknowledgments

- Base model: [`dvitvaai/pothana-base-v2-225M`](https://huggingface.co/dvitvaai/pothana-base-v2-225M)
- Codemix synthetic data: Gemini 2.0 Flash
- Telugu corpus: [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)

## License

Apache 2.0.
