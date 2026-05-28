#!/usr/bin/env python3
"""
Pothana Base 300M — Telugu Text Generation Demo
================================================
Gradio app for the Pothana Base 300M Telugu language model.
Handles raw Telugu text input with automatic Morfessor segmentation.

Developed by Dvitva AI.
"""

import re
import torch
import morfessor
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "dvitvaai/pothana-base-300M"
TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


# ---------------------------------------------------------------------------
# Morfessor segmentation (from inference.py)
# ---------------------------------------------------------------------------
def segment_text(text: str, morf_model, separator: str = "\u2581") -> str:
    """Segment raw text using Morfessor with ▁ word-boundary separators.

    v3: ▁ before each word, bare morphemes (no @@ suffix).
    """
    tokens = text.split()
    seg_tokens = []

    for token in tokens:
        seg_tokens.append(separator)  # ▁ before each word

        if TELUGU_WORD_RE.fullmatch(token):
            segments = morf_model.viterbi_segment(token)[0]
            seg_tokens.extend(segments)
        elif TELUGU_WORD_RE.search(token):
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            parts = [p for p in parts if p]
            for part in parts:
                if TELUGU_WORD_RE.fullmatch(part):
                    segments = morf_model.viterbi_segment(part)[0]
                    seg_tokens.extend(segments)
                else:
                    seg_tokens.append(part)
        else:
            seg_tokens.append(token)

    return " ".join(seg_tokens)


# ---------------------------------------------------------------------------
# Load model, tokenizer, and Morfessor at startup
# ---------------------------------------------------------------------------
print("Loading Morfessor model...")
morf_path = hf_hub_download(repo_id=MODEL_ID, filename="morfessor_telugu.bin")
morf_io = morfessor.MorfessorIO()
morf_model = morf_io.read_binary_model_file(morf_path)

print("Loading model and tokenizer...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=dtype
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Model loaded on {device} ({dtype}). Ready!")


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
):
    """Generate Telugu text from a raw text prompt."""
    if not prompt.strip():
        return ""

    # Step 1: Morfessor segmentation (v3: ▁ separator)
    segmented = segment_text(prompt.strip(), morf_model)

    # Step 2: Tokenize
    inputs = tokenizer(segmented, return_tensors="pt").to(device)

    # Step 3: Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )

    # Step 4: Decode (TeluguTokenizer handles ▁ → space conversion)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
examples = [
    ["తెలుగు భాష చాలా"],
    ["భారతదేశం ఒక"],
    ["ఒకప్పుడు ఒక రాజు"],
    ["హైదరాబాద్ నగరంలో"],
    ["విద్యార్థులు పరీక్షలకు"],
]

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="తెలుగులో టెక్స్ట్ టైప్ చేయండి...",
            lines=3,
        ),
        gr.Slider(
            minimum=50, maximum=500, value=200, step=10,
            label="Max New Tokens",
        ),
        gr.Slider(
            minimum=0.1, maximum=1.5, value=0.8, step=0.05,
            label="Temperature",
        ),
        gr.Slider(
            minimum=10, maximum=100, value=50, step=5,
            label="Top-k",
        ),
        gr.Slider(
            minimum=0.5, maximum=1.0, value=0.95, step=0.05,
            label="Top-p",
        ),
        gr.Slider(
            minimum=1.0, maximum=1.5, value=1.1, step=0.05,
            label="Repetition Penalty",
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=8),
    title="Pothana Base 300M",
    description=(
        "**Telugu text generation** powered by Pothana Base 300M, "
        "a ~300M parameter LLaMA-style model trained from scratch on Telugu text. "
        "Named after [Bammera Pothana](https://en.wikipedia.org/wiki/Bammera_Pothana), "
        "the celebrated 15th-century Telugu poet.\n\n"
        "This is a **base model** (text completion, not chat/instruction-following). "
        "Type a Telugu prompt and the model will continue it.\n\n"
        "Developed by **[Dvitva AI](https://dvitva.ai)**."
    ),
    examples=examples,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
