#!/usr/bin/env python3
"""
Pothana Chat 300M — Telugu Chatbot Demo
========================================
Gradio chatbot for the Pothana Chat 300M Telugu language model.
Handles raw Telugu text input with automatic Morfessor segmentation.

Developed by Dvitva AI.
"""

import re
import torch
import morfessor
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from huggingface_hub import hf_hub_download
from threading import Thread

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "dvitvaai/pothana-chat-300M"
TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")

DEFAULT_SYSTEM = "మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్. వినియోగదారుల ప్రశ్నలకు స్పష్టంగా మరియు సమగ్రంగా సమాధానం ఇవ్వండి."

# Chat special token IDs (must match the model's training)
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
SYSTEM_TOKEN_ID = 86071
USER_TOKEN_ID = 86072
ASSISTANT_TOKEN_ID = 86073
END_TOKEN_ID = 86074


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
# Encode text to token IDs (matching chat.py / SFT training)
# ---------------------------------------------------------------------------
def encode_text(text: str) -> list[int]:
    """Morfessor segment → tokenize to IDs (no BOS/EOS).

    Uses the HF tokenizer but without adding special tokens,
    matching how chat.py and SFT training encode content.
    """
    segmented = segment_text(text, morf_model)
    # Tokenize without BOS (add_special_tokens=False avoids the post-processor
    # which would prepend <bos>)
    ids = tokenizer.encode(segmented, add_special_tokens=False)
    return ids


# ---------------------------------------------------------------------------
# Build chat prompt as token IDs (matching chat.py exactly)
# ---------------------------------------------------------------------------
def build_chat_ids(
    user_message: str,
    history: list[dict],
    system_prompt: str,
) -> list[int]:
    """Build token ID sequence for the chat prompt.

    Constructs IDs directly (like chat.py / SFT training) instead of going
    through apply_chat_template → re-tokenize, which can produce <unk> tokens
    because the HF WordLevel tokenizer lacks char-level fallback.

    Format (matches SFT training):
        <bos> <|system|> {sys_tokens} <|end|>
        <|user|> {ctx_u1} <|end|> <|assistant|> {ctx_a1} <|end|>
        ...
        <|user|> {current} <|end|> <|assistant|>

    Args:
        user_message: Raw Telugu text from the user.
        history: List of {"role": ..., "content": ...} dicts (previous turns).
        system_prompt: System instruction text.

    Returns:
        List of token IDs ready for model input.
    """
    ids = []

    # <bos>
    ids.append(BOS_TOKEN_ID)

    # <|system|> {system_instruction} <|end|>
    ids.append(SYSTEM_TOKEN_ID)
    ids.extend(encode_text(system_prompt))
    ids.append(END_TOKEN_ID)

    # Previous turns from history
    for msg in history:
        content = msg.get("content", "")
        # Handle Gradio 6 structured content: [{"type": "text", "text": "..."}]
        if isinstance(content, list):
            text_parts = [
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = " ".join(text_parts)
        if not isinstance(content, str):
            content = str(content) if content else ""
        if not content.strip():
            continue

        role = msg.get("role", "")
        if role == "user":
            ids.append(USER_TOKEN_ID)
            ids.extend(encode_text(content))
            ids.append(END_TOKEN_ID)
        elif role == "assistant":
            ids.append(ASSISTANT_TOKEN_ID)
            ids.extend(encode_text(content))
            ids.append(END_TOKEN_ID)

    # Current user input
    ids.append(USER_TOKEN_ID)
    ids.extend(encode_text(user_message))
    ids.append(END_TOKEN_ID)

    # Generation starts here
    ids.append(ASSISTANT_TOKEN_ID)

    return ids


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat_respond(
    user_message: str,
    history: list[dict],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
):
    """Generate a streaming chat response.

    Yields partial response text as tokens are generated.
    """
    # Handle Gradio 6 structured message format
    if isinstance(user_message, dict):
        user_message = user_message.get("text", "")
    if not isinstance(user_message, str) or not user_message.strip():
        yield ""
        return

    # Build token IDs directly (matches chat.py / SFT training format)
    prompt_ids = build_chat_ids(user_message, history, system_prompt)

    # Debug: log for troubleshooting
    print(f"[DEBUG] Prompt: {len(prompt_ids)} tokens, first 20: {prompt_ids[:20]}")
    print(f"[DEBUG] History turns: {len(history)}")

    # Convert to tensor
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Set up streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,  # we handle special tokens ourselves
    )

    # Generation kwargs — match chat.py defaults
    # top_p=0 means "off" in our UI; HF needs top_p=1.0 for that
    effective_top_p = float(top_p) if float(top_p) > 0 else 1.0
    temp = float(temperature)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        repetition_penalty=float(repetition_penalty),
        streamer=streamer,
        eos_token_id=[EOS_TOKEN_ID, END_TOKEN_ID],
    )

    # Use greedy decoding when temperature is very low,
    # otherwise sample with the specified parameters
    if temp < 0.01:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temp
        gen_kwargs["top_k"] = int(top_k)
        gen_kwargs["top_p"] = effective_top_p

    # Run generation in a thread so we can stream
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream response tokens
    response = ""
    for text_chunk in streamer:
        # Strip any special tokens that leak through
        text_chunk = text_chunk.replace("<|end|>", "")
        text_chunk = text_chunk.replace("<|assistant|>", "")
        text_chunk = text_chunk.replace("<|user|>", "")
        text_chunk = text_chunk.replace("<|system|>", "")
        text_chunk = text_chunk.replace("<bos>", "")
        text_chunk = text_chunk.replace("<eos>", "")

        if text_chunk:
            response += text_chunk
            yield response

    thread.join()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DESCRIPTION = """\
# Pothana Chat 300M

**Telugu chatbot** powered by Pothana Chat 300M — a ~300M parameter LLaMA-style model \
trained from scratch on Telugu text and instruction-tuned on Telugu conversations.

Named after [Bammera Pothana](https://en.wikipedia.org/wiki/Bammera_Pothana), \
the celebrated 15th-century Telugu poet.

Developed by **[Dvitva AI](https://dvitva.ai)**.

> ⚠️ This is a small research model. Responses may be inaccurate or repetitive. \
Best suited for Telugu language exploration and research.
"""

examples = [
    ["తెలంగాణ రాజధాని ఏది?"],
    ["భారతదేశం గురించి చెప్పండి"],
    ["తెలుగు భాష ప్రత్యేకత ఏమిటి?"],
    ["హైదరాబాద్ లో ప్రసిద్ధ ప్రదేశాలు ఏవి?"],
    ["ఒక చిన్న కథ చెప్పు"],
]

with gr.Blocks(title="Pothana Chat 300M") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Accordion("⚙️ Settings", open=False):
        system_prompt = gr.Textbox(
            value=DEFAULT_SYSTEM,
            label="System Prompt",
            lines=2,
            placeholder="సిస్టమ్ ప్రాంప్ట్ ఇక్కడ టైప్ చేయండి...",
        )
        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=50, maximum=500, value=256, step=10,
                label="Max New Tokens",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                label="Temperature",
            )
        with gr.Row():
            top_k = gr.Slider(
                minimum=10, maximum=100, value=50, step=5,
                label="Top-k",
            )
            top_p = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                label="Top-p (0 = off)",
            )
            repetition_penalty = gr.Slider(
                minimum=1.0, maximum=1.5, value=1.1, step=0.05,
                label="Repetition Penalty",
            )

    chat = gr.ChatInterface(
        fn=chat_respond,
        additional_inputs=[
            system_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        ],
        examples=examples,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
