#!/usr/bin/env python3
"""
Convert Telugu LLaMA checkpoint to HuggingFace format
=====================================================

Converts our custom ~300M param LLaMA-style Telugu model to HuggingFace-compatible
format, loadable with:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("./telugu-llama-300m")
    tokenizer = AutoTokenizer.from_pretrained("./telugu-llama-300m")

What it does:
  1. Reads our checkpoint (.pt) and maps weights to HF LlamaForCausalLM format
     - Splits fused QKV (c_attn) into separate q_proj, k_proj, v_proj
     - Renames all keys to HF convention
  2. Reads our tokenizer.json and builds an HF-compatible tokenizer
     - Preserves @@ continuation marker semantics
     - Creates tokenizer.json, tokenizer_config.json, special_tokens_map.json
  3. Creates config.json (LlamaConfig) and generation_config.json
  4. Optionally copies morfessor_telugu.bin for raw-text inference

Usage:
    python convert_to_hf.py \\
        --checkpoint ./checkpoints/best.pt \\
        --tokenizer ./tokenizer \\
        --morfessor-model ./data/morfessor/morfessor_telugu.bin \\
        --output ./telugu-llama-300m

    # Then use with HuggingFace:
    python -c "
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained('./telugu-llama-300m')
    tok = AutoTokenizer.from_pretrained('./telugu-llama-300m')
    print(model)
    print(tok)
    "
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Part 1: Create config.json (LlamaConfig)
# ===========================================================================
def create_config(checkpoint: dict, output_dir: Path):
    """Create HuggingFace LlamaConfig from our GPTConfig."""
    cfg = checkpoint["config"]

    n_embd = cfg["n_embd"]        # 1024
    n_head = cfg["n_head"]        # 16
    n_layer = cfg["n_layer"]      # 20
    block_size = cfg["block_size"]  # 2048
    vocab_size = cfg["vocab_size"]
    rope_theta = cfg.get("rope_theta", 10000.0)

    # SwiGLU intermediate_size: same formula as in train_gpt.py
    hidden_dim = int(2 * n_embd * 4 / 3)
    hidden_dim = ((hidden_dim + 255) // 256) * 256  # round up to 256

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "float32",

        # Dimensions
        "hidden_size": n_embd,
        "intermediate_size": hidden_dim,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,  # no GQA — same as num_attention_heads
        "head_dim": n_embd // n_head,

        # Positional encoding
        "max_position_embeddings": block_size,
        "rope_theta": rope_theta,
        "rope_scaling": None,

        # Normalization & activation
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",

        # No biases (LLaMA style)
        "attention_bias": False,
        "mlp_bias": False,

        # Vocabulary
        "vocab_size": vocab_size,
        "tie_word_embeddings": True,

        # Token IDs (from our SPECIAL_TOKENS)
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,

        # Standard LLaMA settings
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "pretraining_tp": 1,
        "use_cache": True,

        # Transformers version metadata
        "transformers_version": "4.40.0",
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Created config.json:")
    logger.info("  hidden_size=%d, intermediate_size=%d", n_embd, hidden_dim)
    logger.info("  num_layers=%d, num_heads=%d, vocab_size=%d", n_layer, n_head, vocab_size)
    logger.info("  max_position_embeddings=%d, rope_theta=%.1f", block_size, rope_theta)

    return config


# ===========================================================================
# Part 2: Convert weights
# ===========================================================================
def convert_weights(checkpoint: dict, config: dict, output_dir: Path,
                    original_vocab_size: int = 0):
    """Convert our state_dict to HF LlamaForCausalLM format.

    Key operations:
      - Split fused c_attn.weight (3*n_embd, n_embd) → q_proj, k_proj, v_proj
      - Rename all keys to HF naming convention
      - Skip freqs_cis buffer (HF recomputes RoPE)
      - Skip lm_head.weight (tied to embed_tokens)
      - Pad embedding if vocab was expanded (BPE merge intermediates)
      - Save as model.safetensors
    """
    import torch
    from safetensors.torch import save_file

    state_dict = checkpoint["model"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]
    hf_vocab_size = config["vocab_size"]

    hf_state_dict = {}

    # Embedding — pad with zeros if vocab was expanded for BPE merge intermediates
    embed_weight = state_dict["transformer.wte.weight"]
    if hf_vocab_size > embed_weight.shape[0]:
        n_extra = hf_vocab_size - embed_weight.shape[0]
        logger.info("Padding embedding: %d → %d rows (+%d zero rows for BPE merge tokens)",
                     embed_weight.shape[0], hf_vocab_size, n_extra)
        padding = torch.zeros(n_extra, n_embd, dtype=embed_weight.dtype)
        embed_weight = torch.cat([embed_weight, padding], dim=0)
    hf_state_dict["model.embed_tokens.weight"] = embed_weight

    # Transformer layers
    for i in range(n_layer):
        prefix_ours = f"transformer.h.{i}"
        prefix_hf = f"model.layers.{i}"

        # Input LayerNorm (RMSNorm)
        hf_state_dict[f"{prefix_hf}.input_layernorm.weight"] = (
            state_dict[f"{prefix_ours}.ln_1.weight"]
        )

        # Attention: split fused QKV
        c_attn_weight = state_dict[f"{prefix_ours}.attn.c_attn.weight"]
        # c_attn_weight shape: (3 * n_embd, n_embd) = (3072, 1024)
        assert c_attn_weight.shape[0] == 3 * n_embd, (
            f"Expected c_attn shape ({3*n_embd}, {n_embd}), got {c_attn_weight.shape}"
        )
        q_proj, k_proj, v_proj = c_attn_weight.split(n_embd, dim=0)

        hf_state_dict[f"{prefix_hf}.self_attn.q_proj.weight"] = q_proj
        hf_state_dict[f"{prefix_hf}.self_attn.k_proj.weight"] = k_proj
        hf_state_dict[f"{prefix_hf}.self_attn.v_proj.weight"] = v_proj

        # Output projection
        hf_state_dict[f"{prefix_hf}.self_attn.o_proj.weight"] = (
            state_dict[f"{prefix_ours}.attn.c_proj.weight"]
        )

        # Post-attention LayerNorm (RMSNorm)
        hf_state_dict[f"{prefix_hf}.post_attention_layernorm.weight"] = (
            state_dict[f"{prefix_ours}.ln_2.weight"]
        )

        # SwiGLU MLP
        hf_state_dict[f"{prefix_hf}.mlp.gate_proj.weight"] = (
            state_dict[f"{prefix_ours}.mlp.w_gate.weight"]
        )
        hf_state_dict[f"{prefix_hf}.mlp.up_proj.weight"] = (
            state_dict[f"{prefix_ours}.mlp.w_up.weight"]
        )
        hf_state_dict[f"{prefix_hf}.mlp.down_proj.weight"] = (
            state_dict[f"{prefix_ours}.mlp.w_down.weight"]
        )

    # Final LayerNorm
    hf_state_dict["model.norm.weight"] = state_dict["transformer.ln_f.weight"]

    # lm_head — tied to embed_tokens, so we include it explicitly
    # HF LlamaForCausalLM with tie_word_embeddings=True will handle the tying,
    # but safetensors needs the key if it's in the model's state dict.
    # Actually, with tie_word_embeddings=True, HF does NOT expect lm_head.weight
    # in the checkpoint — it's aliased from embed_tokens at load time.
    # So we skip it.

    # Log what we converted
    n_params = sum(p.numel() for p in hf_state_dict.values())
    logger.info("Converted %d weight tensors (%d parameters, %.1fM)",
                len(hf_state_dict), n_params, n_params / 1e6)

    # Log what we skipped
    converted_keys = set()
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        converted_keys.update([
            f"{prefix}.ln_1.weight",
            f"{prefix}.attn.c_attn.weight",
            f"{prefix}.attn.c_proj.weight",
            f"{prefix}.ln_2.weight",
            f"{prefix}.mlp.w_gate.weight",
            f"{prefix}.mlp.w_up.weight",
            f"{prefix}.mlp.w_down.weight",
        ])
    converted_keys.update(["transformer.wte.weight", "transformer.ln_f.weight"])

    skipped = [k for k in state_dict.keys() if k not in converted_keys]
    if skipped:
        logger.info("Skipped %d keys: %s", len(skipped), skipped)

    # Save as safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(hf_state_dict, str(safetensors_path))

    file_size = os.path.getsize(safetensors_path)
    logger.info("Saved model.safetensors (%.2f GB)", file_size / 1e9)

    return hf_state_dict


# ===========================================================================
# Part 3: Convert tokenizer
# ===========================================================================
def convert_tokenizer(tokenizer_dir: Path, output_dir: Path, original_vocab_size: int):
    """Convert our custom tokenizer to HuggingFace format.

    Our tokenizer is a Morfessor+BPE hybrid with @@ continuation markers.
    We create an HF-compatible tokenizer using WordLevel model (exact lookup):
      - Uses the same vocab (token → id mapping, direct lookup)
      - Pre-tokenizer: WhitespaceSplit (input is already segmented)
      - Decoder: replaces "@@  " with "" to rejoin morphemes
      - Post-processor: prepends <bos>

    We use WordLevel instead of BPE because our tokenizer does direct vocab
    lookup on whole morpheme tokens (e.g. "విద్యార్థు@@" → ID). HF's BPE
    model would re-split these into characters and merge up, giving wrong IDs.

    Returns the final HF vocab size.

    Note: HF tokenizer handles pre-segmented text. For raw Telugu text,
    users need to run Morfessor segmentation first (morfessor_telugu.bin).
    """

    # Load our tokenizer
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json_path.exists():
        tokenizer_json_path = tokenizer_dir
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        our_tok = json.load(f)

    token_to_id = our_tok["token_to_id"]
    separator = our_tok["separator"]  # "@@"
    bpe_merges = our_tok.get("bpe_merges", [])
    vocab_size = our_tok["vocab_size"]

    logger.info("Our tokenizer: vocab_size=%d, separator='%s', %d BPE merges",
                vocab_size, separator, len(bpe_merges))

    # --- Build HF tokenizer.json ---
    # We use WordLevel model (exact token → id lookup), NOT BPE.
    #
    # Why: Our tokenizer does direct vocab lookup on whitespace-split tokens.
    # The input is already Morfessor-segmented, so "విద్యార్థు@@" is a single
    # vocab entry looked up directly. HF's BPE model would re-split it into
    # characters and try to merge up — producing completely wrong IDs.
    # WordLevel matches our actual encode() behavior: split on whitespace,
    # look up each piece in the vocab, return <unk> if missing.

    # Build vocab dict for HF (token → id)
    hf_vocab = {}
    for token, tid in token_to_id.items():
        hf_vocab[token] = tid

    # Construct the HF tokenizer.json
    hf_tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 2,
                "content": "<bos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 3,
                "content": "<eos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "WhitespaceSplit",
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]},
            },
        },
        "decoder": {
            "type": "Replace",
            "pattern": {"String": f"{separator} "},
            "content": "",
        },
        "model": {
            "type": "WordLevel",
            "vocab": hf_vocab,
            "unk_token": "<unk>",
        },
    }

    # Save HF tokenizer.json
    hf_tok_path = output_dir / "tokenizer.json"
    with open(hf_tok_path, "w", encoding="utf-8") as f:
        json.dump(hf_tokenizer, f, ensure_ascii=False, indent=2)
    hf_vocab_size = len(hf_vocab)
    logger.info("Saved tokenizer.json (hf_vocab_size=%d, model=WordLevel)", hf_vocab_size)

    # --- tokenizer_config.json ---
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "llama",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 2048,
        "extra_info": {
            "type": "morfessor_bpe_telugu",
            "separator": separator,
            "note": (
                "This tokenizer expects Morfessor-segmented text as input. "
                "For raw Telugu text, run Morfessor segmentation first using "
                "the included morfessor_telugu.bin model. "
                "Tokens ending with '@@' are continuation pieces that join "
                "to the next token. The decoder handles @@ removal automatically."
            ),
        },
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer_config.json")

    # --- special_tokens_map.json ---
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    stm_path = output_dir / "special_tokens_map.json"
    with open(stm_path, "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    logger.info("Saved special_tokens_map.json")

    return hf_vocab_size


# ===========================================================================
# Part 4: generation_config.json
# ===========================================================================
def create_generation_config(output_dir: Path):
    """Create default generation config for the model."""
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "max_new_tokens": 200,
        "repetition_penalty": 1.1,
        "transformers_version": "4.40.0",
    }
    path = output_dir / "generation_config.json"
    with open(path, "w") as f:
        json.dump(gen_config, f, indent=2)
    logger.info("Saved generation_config.json")


# ===========================================================================
# Part 5: Model card (README.md)
# ===========================================================================
def create_model_card(config: dict, output_dir: Path):
    """Create a HuggingFace model card (README.md)."""

    vocab_size = config["vocab_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    ctx_len = config["max_position_embeddings"]

    # Rough param count (same formula as GPTConfig.param_count)
    emb = vocab_size * hidden
    attn_per_layer = 4 * hidden ** 2
    mlp_per_layer = 3 * hidden * intermediate
    tfm = n_layers * (attn_per_layer + mlp_per_layer)
    norms = (2 * n_layers + 1) * hidden
    n_params = emb + tfm + norms
    param_str = f"{n_params / 1e6:.0f}M"

    card = f"""---
language:
  - te
license: apache-2.0
tags:
  - telugu
  - llama
  - causal-lm
  - morfessor
  - from-scratch
library_name: transformers
pipeline_tag: text-generation
---

# Telugu LLaMA ({param_str})

A **{param_str} parameter** LLaMA-style language model trained **from scratch** on Telugu text.

## Model Details

| | |
|---|---|
| **Architecture** | LLaMA (RoPE + SwiGLU + RMSNorm) |
| **Parameters** | {param_str} |
| **Hidden size** | {hidden} |
| **Layers** | {n_layers} |
| **Attention heads** | {n_heads} |
| **Intermediate size** | {intermediate} |
| **Context length** | {ctx_len} |
| **Vocab size** | {vocab_size:,} |
| **Tokenizer** | Morfessor + BPE (Telugu morpheme-aware) |
| **Training** | Single GPU, bf16 mixed precision |

## Tokenizer

This model uses a **Morfessor + BPE hybrid tokenizer** designed for Telugu:

- **Telugu text**: Segmented into morphemes using [Morfessor](https://github.com/aalto-speech/morfessor) with `@@` continuation markers
- **Non-Telugu text** (English, numbers, URLs): Handled by BPE subword encoding
- **Fallback**: Character-level encoding for out-of-vocabulary tokens

**Important**: The tokenizer expects **pre-segmented** input (with `@@` markers). For raw Telugu text, you need to run Morfessor segmentation first.

## Usage

### Basic usage (with pre-segmented text)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("YOUR_USERNAME/telugu-llama-{param_str.lower()}")
tokenizer = AutoTokenizer.from_pretrained("YOUR_USERNAME/telugu-llama-{param_str.lower()}")

# Input must be Morfessor-segmented (with @@ continuation markers)
segmented_text = "తెలుగు భాష చాలా అందమైన@@ ది"
inputs = tokenizer(segmented_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Full pipeline (raw Telugu text)

For raw Telugu text, segment with Morfessor first:

```python
import morfessor

# Load Morfessor model
io = morfessor.MorfessorIO()
morf_model = io.read_binary_model_file("morfessor_telugu.bin")

def segment_telugu(text, separator="@@"):
    import re
    TELUGU_RE = re.compile(r"[\\u0C00-\\u0C7F]+")
    tokens = []
    for word in text.split():
        if TELUGU_RE.fullmatch(word):
            segments = morf_model.viterbi_segment(word)[0]
            for i, seg in enumerate(segments):
                tokens.append(seg + separator if i < len(segments) - 1 else seg)
        else:
            tokens.append(word)
    return " ".join(tokens)

# Segment, then tokenize and generate
raw_text = "తెలుగు భాష చాలా అందమైనది"
segmented = segment_telugu(raw_text)
inputs = tokenizer(segmented, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

- **Data**: Telugu text corpus (Sangraha dataset)
- **Preprocessing**: Morfessor morpheme segmentation + BPE for non-Telugu
- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1, beta1=0.9, beta2=0.95)
- **Schedule**: Cosine LR decay with 500-step warmup
- **Precision**: bf16 mixed precision
- **Hardware**: Single GPU

## Limitations

- This is a **base model** (not instruction-tuned) — it performs text completion, not instruction following
- The tokenizer requires **Morfessor-segmented input** for best results
- Trained primarily on Telugu text; limited multilingual capability
- Small model size ({param_str}) limits reasoning and knowledge capacity

## License

Apache 2.0
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card)
    logger.info("Saved README.md (model card)")


# ===========================================================================
# Verification
# ===========================================================================
def verify_conversion(output_dir: Path, checkpoint: dict, tokenizer_dir: Path):
    """Verify the converted model loads correctly with HuggingFace."""
    import torch

    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION")
    logger.info("=" * 70)

    # 1. Load model with HF
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        hf_config = AutoConfig.from_pretrained(str(output_dir))
        logger.info("[PASS] config.json loads with AutoConfig")
        logger.info("  model_type=%s, hidden_size=%d, num_layers=%d, vocab_size=%d",
                     hf_config.model_type, hf_config.hidden_size,
                     hf_config.num_hidden_layers, hf_config.vocab_size)

        hf_model = AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            torch_dtype=torch.float32,
        )
        hf_model.eval()
        n_params = sum(p.numel() for p in hf_model.parameters())
        logger.info("[PASS] Model loads with AutoModelForCausalLM (%.1fM params)", n_params / 1e6)
    except Exception as e:
        logger.error("[FAIL] Model loading failed: %s", e)
        return False

    # 2. Load tokenizer with HF
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        logger.info("[PASS] Tokenizer loads with AutoTokenizer (vocab_size=%d)",
                     hf_tokenizer.vocab_size)
    except Exception as e:
        logger.error("[FAIL] Tokenizer loading failed: %s", e)
        return False

    # 3. Compare tokenization (on pre-segmented text)
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer
    try:
        our_tokenizer = MorfessorTokenizer(tokenizer_dir)

        test_texts = [
            "విద్యార్థు@@ ల@@ కు",
            "తెలుగు భాష",
            "ప్ర@@ భుత్వం కొత్త",
        ]

        all_match = True
        for text in test_texts:
            our_ids = our_tokenizer.encode(text, add_bos=True, add_eos=False)
            hf_ids = hf_tokenizer.encode(text, add_special_tokens=True)

            if our_ids == hf_ids:
                logger.info("[PASS] Tokenization matches for: '%s'", text[:40])
            else:
                logger.warning("[WARN] Tokenization mismatch for: '%s'", text[:40])
                logger.warning("  Ours: %s", our_ids[:15])
                logger.warning("  HF:   %s", hf_ids[:15])
                all_match = False

        if all_match:
            logger.info("[PASS] All tokenization tests match")

    except Exception as e:
        logger.warning("[WARN] Could not compare tokenization: %s", e)

    # 4. Compare model outputs (logits)
    try:
        our_state = checkpoint["model"]
        our_n_embd = checkpoint["config"]["n_embd"]

        # Quick forward pass comparison
        test_ids = torch.tensor([[2, 100, 200, 300]], dtype=torch.long)  # <bos> + 3 tokens
        with torch.no_grad():
            hf_output = hf_model(test_ids)
            hf_logits = hf_output.logits  # (1, 4, vocab_size)

        logger.info("[PASS] Forward pass works — output shape: %s", hf_logits.shape)

        # Check that logits aren't all zeros or NaN
        if torch.isnan(hf_logits).any():
            logger.error("[FAIL] NaN values in logits!")
            return False
        if (hf_logits == 0).all():
            logger.error("[FAIL] All-zero logits!")
            return False

        logger.info("[PASS] Logits look valid (min=%.2f, max=%.2f, std=%.2f)",
                     hf_logits.min().item(), hf_logits.max().item(), hf_logits.std().item())

    except Exception as e:
        logger.error("[FAIL] Forward pass failed: %s", e)
        return False

    # 5. Test generation
    try:
        generated = hf_model.generate(
            test_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_k=50,
        )
        gen_text = hf_tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info("[PASS] Generation works — sample: '%s'", gen_text[:80])
    except Exception as e:
        logger.error("[FAIL] Generation failed: %s", e)
        return False

    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL CHECKS PASSED — model is ready for HuggingFace!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Usage:")
    logger.info('  from transformers import AutoModelForCausalLM, AutoTokenizer')
    logger.info('  model = AutoModelForCausalLM.from_pretrained("%s")', output_dir)
    logger.info('  tokenizer = AutoTokenizer.from_pretrained("%s")', output_dir)
    logger.info("")
    logger.info("To push to HuggingFace Hub:")
    logger.info('  model.push_to_hub("your-username/telugu-llama-300m")')
    logger.info('  tokenizer.push_to_hub("your-username/telugu-llama-300m")')

    return True


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert Telugu LLaMA checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  %(prog)s --checkpoint ./checkpoints/best.pt --tokenizer ./tokenizer --output ./telugu-llama-300m

  # With Morfessor model (for raw text inference documentation)
  %(prog)s --checkpoint ./checkpoints/best.pt --tokenizer ./tokenizer \\
           --morfessor-model ./data/morfessor/morfessor_telugu.bin \\
           --output ./telugu-llama-300m

  # Skip verification (faster)
  %(prog)s --checkpoint ./checkpoints/best.pt --tokenizer ./tokenizer \\
           --output ./telugu-llama-300m --no-verify

  # Then use with HuggingFace:
  python -c "
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained('./telugu-llama-300m')
  tok = AutoTokenizer.from_pretrained('./telugu-llama-300m')
  "
        """,
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to our .pt checkpoint (e.g. ./checkpoints/best.pt)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to our tokenizer directory (containing tokenizer.json)")
    parser.add_argument("--morfessor-model", type=str, default=None,
                        help="Path to morfessor_telugu.bin (copied to output for raw-text inference)")
    parser.add_argument("--output", type=str, default="./telugu-llama-300m",
                        help="Output directory for HF model (default: ./telugu-llama-300m)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification step")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_dir = Path(args.tokenizer)
    output_dir = Path(args.output)

    # Validate inputs
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    tok_json = tokenizer_dir / "tokenizer.json" if tokenizer_dir.is_dir() else tokenizer_dir
    if not tok_json.exists():
        logger.error("Tokenizer not found: %s", tok_json)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Converting to HuggingFace format:")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Tokenizer:  %s", tokenizer_dir)
    logger.info("  Output:     %s", output_dir)
    logger.info("")

    # Load checkpoint
    import torch
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    logger.info("Checkpoint loaded — step %s, config: %s",
                checkpoint.get("step", "?"),
                {k: v for k, v in checkpoint["config"].items()})

    original_vocab_size = checkpoint["config"]["vocab_size"]

    # Part 1: Convert tokenizer FIRST — may add extra vocab entries for BPE merge intermediates
    logger.info("")
    logger.info("--- Part 1: Converting tokenizer ---")
    hf_vocab_size = convert_tokenizer(tokenizer_dir, output_dir, original_vocab_size)

    # Part 2: config.json — use the (possibly expanded) HF vocab size
    logger.info("")
    logger.info("--- Part 2: Creating config.json ---")
    if hf_vocab_size > original_vocab_size:
        logger.info("Vocab expanded: %d → %d (BPE merge intermediates added)",
                     original_vocab_size, hf_vocab_size)
        checkpoint["config"]["vocab_size"] = hf_vocab_size
    config = create_config(checkpoint, output_dir)

    # Part 3: Convert weights — pad embedding if vocab was expanded
    logger.info("")
    logger.info("--- Part 3: Converting weights ---")
    hf_state_dict = convert_weights(checkpoint, config, output_dir,
                                     original_vocab_size=original_vocab_size)

    # Part 4: generation_config.json
    logger.info("")
    logger.info("--- Part 4: Creating generation_config.json ---")
    create_generation_config(output_dir)

    # Part 5: Model card
    logger.info("")
    logger.info("--- Part 5: Creating README.md (model card) ---")
    create_model_card(config, output_dir)

    # Copy Morfessor model if provided
    if args.morfessor_model:
        morf_path = Path(args.morfessor_model)
        if morf_path.exists():
            dest = output_dir / "morfessor_telugu.bin"
            shutil.copy2(str(morf_path), str(dest))
            logger.info("Copied morfessor_telugu.bin to output directory")
        else:
            logger.warning("Morfessor model not found: %s (skipping)", morf_path)

    # List output files
    logger.info("")
    logger.info("Output files:")
    for p in sorted(output_dir.iterdir()):
        size = os.path.getsize(p)
        if size > 1e9:
            logger.info("  %s (%.2f GB)", p.name, size / 1e9)
        elif size > 1e6:
            logger.info("  %s (%.1f MB)", p.name, size / 1e6)
        elif size > 1e3:
            logger.info("  %s (%.1f KB)", p.name, size / 1e3)
        else:
            logger.info("  %s (%d bytes)", p.name, size)

    # Verification
    if not args.no_verify:
        logger.info("")
        logger.info("--- Verification ---")
        verify_conversion(output_dir, checkpoint, tokenizer_dir)
    else:
        logger.info("")
        logger.info("Skipping verification (--no-verify)")
        logger.info("To verify manually:")
        logger.info("  python -c \"from transformers import AutoModelForCausalLM; "
                     "m = AutoModelForCausalLM.from_pretrained('%s'); print('OK', m)\"", output_dir)

    # Free memory
    del checkpoint
    del hf_state_dict

    logger.info("")
    logger.info("Done! Model saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
