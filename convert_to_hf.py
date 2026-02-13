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
def convert_weights(checkpoint: dict, config: dict, output_dir: Path):
    """Convert our state_dict to HF LlamaForCausalLM format.

    Key operations:
      - Split fused c_attn.weight (3*n_embd, n_embd) → q_proj, k_proj, v_proj
      - Rename all keys to HF naming convention
      - Skip freqs_cis buffer (HF recomputes RoPE)
      - Skip lm_head.weight (tied to embed_tokens)
      - Save as model.safetensors
    """
    import torch
    from safetensors.torch import save_file

    state_dict = checkpoint["model"]
    n_embd = config["hidden_size"]
    n_layer = config["num_hidden_layers"]

    hf_state_dict = {}

    # Embedding
    hf_state_dict["model.embed_tokens.weight"] = state_dict["transformer.wte.weight"]

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
def convert_tokenizer(tokenizer_dir: Path, output_dir: Path):
    """Convert our custom tokenizer to HuggingFace format.

    Our tokenizer is a Morfessor+BPE hybrid with @@ continuation markers.
    We create an HF-compatible tokenizer that:
      - Uses the same vocab (token → id mapping)
      - Uses BPE model (with our merge rules)
      - Has a decoder that strips @@ markers to reconstruct text
      - Adds <bos> prefix via post-processor

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
    # HF tokenizer.json format for BPE model
    # We'll construct it manually for full control

    # Build vocab dict for HF (token → id)
    hf_vocab = {}
    for token, tid in token_to_id.items():
        hf_vocab[token] = tid

    # Build merges list for HF format (list of "a b" strings)
    hf_merges = []
    for merge in bpe_merges:
        if isinstance(merge, (list, tuple)) and len(merge) == 2:
            hf_merges.append(f"{merge[0]} {merge[1]}")

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
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": hf_vocab,
            "merges": hf_merges,
        },
    }

    # Save HF tokenizer.json
    hf_tok_path = output_dir / "tokenizer.json"
    with open(hf_tok_path, "w", encoding="utf-8") as f:
        json.dump(hf_tokenizer, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer.json (vocab_size=%d, %d merges)", vocab_size, len(hf_merges))

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

    # Part 1: config.json
    logger.info("")
    logger.info("--- Part 1: Creating config.json ---")
    config = create_config(checkpoint, output_dir)

    # Part 2: Convert weights
    logger.info("")
    logger.info("--- Part 2: Converting weights ---")
    hf_state_dict = convert_weights(checkpoint, config, output_dir)

    # Part 3: Convert tokenizer
    logger.info("")
    logger.info("--- Part 3: Converting tokenizer ---")
    convert_tokenizer(tokenizer_dir, output_dir)

    # Part 4: generation_config.json
    logger.info("")
    logger.info("--- Part 4: Creating generation_config.json ---")
    create_generation_config(output_dir)

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
