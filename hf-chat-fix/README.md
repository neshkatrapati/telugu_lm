---
language:
  - te
license: apache-2.0
tags:
  - telugu
  - llama
  - causal-lm
  - chat
  - sft
  - instruction-tuned
  - morfessor
  - from-scratch
library_name: transformers
pipeline_tag: text-generation
base_model: dvitvaai/pothana-base-300M
---

# Pothana Chat 300M

A **345M parameter** LLaMA-style chat model for Telugu, instruction-tuned from [Pothana Base 300M](https://huggingface.co/dvitvaai/pothana-base-300M).

Named after [Bammera Pothana](https://en.wikipedia.org/wiki/Bammera_Pothana), the celebrated 15th-century Telugu poet who authored the *Andhra Maha Bhagavatamu*.

Developed by **[Dvitva AI](https://dvitva.ai)**.

## Model Details

| | |
|---|---|
| **Model** | pothana-chat-300M |
| **Base model** | [pothana-base-300M](https://huggingface.co/dvitvaai/pothana-base-300M) |
| **Architecture** | LLaMA (RoPE + SwiGLU + RMSNorm) |
| **Parameters** | 345M |
| **Hidden size** | 1024 |
| **Layers** | 20 |
| **Attention heads** | 16 |
| **Intermediate size** | 2816 |
| **Context length** | 2048 |
| **Vocab size** | 86,075 (base + 4 chat tokens) |
| **Tokenizer** | Morfessor + BPE (Telugu morpheme-aware) |
| **Fine-tuning** | Full SFT on Telugu conversations |
| **Developed by** | [Dvitva AI](https://dvitva.ai) |

## Chat Template

This model uses the following chat format (matching its SFT training):

```
<bos><|system|> {system instruction} <|end|><|user|> {user message} <|end|><|assistant|> {response} <|end|>
```

### Multi-turn example

```
<bos><|system|> మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్. <|end|>
<|user|> తెలంగాణ రాజధాని ఏది? <|end|>
<|assistant|> తెలంగాణ రాజధాని హైదరాబాద్. <|end|>
<|user|> దాని జనాభా ఎంత? <|end|>
<|assistant|>
```

The model generates after `<|assistant|>` and stops at `<|end|>`.

### Special Tokens

| Token | ID |
|---|---|
| `<|system|>` | 86071 |
| `<|user|>` | 86072 |
| `<|assistant|>` | 86073 |
| `<|end|>` | 86074 |
| `<bos>` | 2 |
| `<eos>` | 3 |
| `<pad>` | 0 |

## Quick Start

### Using the chat template (recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("dvitvaai/pothana-chat-300M")
tokenizer = AutoTokenizer.from_pretrained("dvitvaai/pothana-chat-300M", trust_remote_code=True)

messages = [
    {"role": "system", "content": "మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్."},
    {"role": "user", "content": "తెలంగాణ రాజధాని ఏది?"},
]

# Apply chat template (handles formatting automatically)
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

> **Note**: `trust_remote_code=True` is required for the custom tokenizer that handles `@@` morpheme joining.

### Manual prompt construction

If you prefer to build the prompt manually:

```python
# For Morfessor-segmented text:
prompt = "<bos><|system|> మీరు ఒక సహాయ@@ కరమైన తెలుగు AI అసిస్టెంట్. <|end|><|user|> తెలం@@ గాణ రాజ@@ ధాని ఏది? <|end|><|assistant|>"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### Using the CLI chat script

For the best interactive experience, use the `chat.py` CLI from the [training repo](https://github.com/dvitvaai/telugu-lm):

```bash
# Interactive multi-turn chat
python chat.py --checkpoint ./sft_checkpoints/best.pt

# Single question
python chat.py -c ./sft_checkpoints/best.pt -p "తెలంగాణ రాజధాని ఏది?"

# With custom system instruction
python chat.py -c ./sft_checkpoints/best.pt \
    --system "మీరు తెలుగు సహాయకుడు. సంక్షిప్తంగా సమాధానం ఇవ్వండి."
```

The CLI supports: streaming output, KV-cache for fast generation, multi-turn context, adjustable temperature/top-k/top-p, and in-session commands (`/reset`, `/config`, `/set`, `/history`).

## Tokenizer

This model uses a **Morfessor + BPE hybrid tokenizer** designed for Telugu:

- **Telugu text**: Segmented into morphemes using [Morfessor](https://github.com/aalto-speech/morfessor) with `@@` continuation markers
- **Non-Telugu text** (English, numbers): Handled by BPE subword encoding
- **Fallback**: Character-level encoding for out-of-vocabulary tokens

**Important**: The tokenizer expects **pre-segmented** input (with `@@` markers). For raw Telugu text, you need to run Morfessor segmentation first using the included `morfessor_telugu.bin`.

### Full pipeline (raw Telugu text)

```python
import morfessor, re
from huggingface_hub import hf_hub_download

# Load Morfessor model from the repo
morf_path = hf_hub_download(repo_id="dvitvaai/pothana-chat-300M", filename="morfessor_telugu.bin")
io = morfessor.MorfessorIO()
morf_model = io.read_binary_model_file(morf_path)

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]+")

def segment_telugu(text, separator="@@"):
    tokens = []
    for word in text.split():
        if TELUGU_RE.fullmatch(word):
            segments = morf_model.viterbi_segment(word)[0]
            for i, seg in enumerate(segments):
                tokens.append(seg + separator if i < len(segments) - 1 else seg)
        else:
            tokens.append(word)
    return " ".join(tokens)

# Segment, format as chat, tokenize, generate
raw_text = "భారతదేశ ప్రధానమంత్రి ఎవరు?"
segmented = segment_telugu(raw_text)

messages = [
    {"role": "system", "content": segment_telugu("మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్.")},
    {"role": "user", "content": segmented},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Base model**: [Pothana Base 300M](https://huggingface.co/dvitvaai/pothana-base-300M) (pretrained on Telugu text corpus)
- **Fine-tuning method**: Full SFT (all parameters updated)
- **Data**: ~1000 multi-turn Telugu conversations (~6,700 turn pairs, windowed into ~6,700 training examples)
- **Chat template**: System instruction + multi-turn context + user/assistant turns
- **Loss masking**: Only assistant response tokens contribute to loss
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Cosine LR decay with 50-step warmup
- **Effective batch size**: 16 (8 micro-batch x 2 gradient accumulation)
- **Precision**: bf16 mixed precision
- **Early stopping**: Patience 3 on validation loss

## Limitations

- **Small model** (345M) — limited reasoning and knowledge capacity
- **Limited training data** (~1000 conversations) — may struggle with topics not covered in training
- **Requires Morfessor segmentation** — raw Telugu text must be segmented before tokenization
- **Telugu-only** — primarily trained on Telugu; limited multilingual capability
- **May hallucinate** — as with all small LMs, responses may contain inaccurate information

## License

Apache 2.0

## Citation

```
@misc{pothana-chat-300M,
  title={Pothana Chat 300M: A Telugu Chat Language Model},
  author={Dvitva AI},
  year={2025},
  url={https://huggingface.co/dvitvaai/pothana-chat-300M}
}
```
