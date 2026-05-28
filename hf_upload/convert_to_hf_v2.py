#!/usr/bin/env python3
"""
Convert Base v2 checkpoint to HuggingFace format.

Inputs:
  --checkpoint  ./checkpoints/base-v2/final.pt
  --tokenizer   ./tokenizer-new-v5
  --morfessor   ./data/morfessor/morfessor_telugu.bin
  --output      ./hf_release/pothana-base-v2-225M

Outputs (in --output dir):
  config.json                  PothanaConfig (use_qk_norm=True, tie_word_embeddings=False)
  generation_config.json
  model.safetensors            ~890 MB (fp32). Untied lm_head + duplicated layers for weight sharing.
  modeling_pothana.py          Custom modeling code (PothanaForCausalLM)
  tokenizer_class.py           PothanaTokenizer (handles @@ continuation prefix)
  tokenizer.json               HF tokenizers WordLevel
  tokenizer_config.json        auto_map → PothanaTokenizer
  special_tokens_map.json
  morfessor_telugu.bin         For raw-text preprocessing

Loads cleanly via:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
  tok = AutoTokenizer.from_pretrained(..., trust_remote_code=True)
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _llama_rope_permute_2d(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Permute the output rows of a [n_heads*head_dim, in_dim] projection from
    interleaved (complex-pair) RoPE layout to HF Llama's halved (rotate_half) layout.

    Within each head's head_dim rows: even rows come first, then odd.
        old: [q0, q1, q2, q3, ..., q46, q47]
        new: [q0, q2, q4, ..., q46, q1, q3, ..., q47]
    """
    return w.reshape(n_heads, head_dim // 2, 2, -1).transpose(1, 2).reshape(n_heads * head_dim, -1)


def _llama_rope_permute_1d(w: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Same permutation, for a [head_dim] norm weight (e.g. q_norm)."""
    return w.reshape(head_dim // 2, 2).transpose(0, 1).reshape(head_dim)


def remap_state_dict(sd: dict, n_unique_layers: int, ws: bool,
                     n_head: int, n_kv_head: int, head_dim: int) -> dict:
    """Map our GPT state_dict keys to HF Llama naming + permute Q/K for RoPE convention.

    Our train_gpt.py uses interleaved (complex-pair) RoPE; HF Llama uses
    halved (rotate_half). Q/K projections and Q/K-norm weights need permutation
    of their output dim to bridge conventions.
    """
    new = {}

    # Top-level
    if "transformer.wte.weight" in sd:
        new["model.embed_tokens.weight"] = sd["transformer.wte.weight"]
    if "lm_head.weight" in sd:
        new["lm_head.weight"] = sd["lm_head.weight"]
    if "transformer.ln_f.weight" in sd:
        new["model.norm.weight"] = sd["transformer.ln_f.weight"]

    # Per-block. Each entry: (src_suffix, dst_suffix, rope_permute_kind)
    # rope_permute_kind in {None, ("2d", n_heads_for_dim), ("1d",)}
    PER_BLOCK_MAP = [
        ("ln_1.weight",          "input_layernorm.weight",           None),
        ("ln_2.weight",          "post_attention_layernorm.weight",  None),
        ("attn.q_proj.weight",   "self_attn.q_proj.weight",          ("2d", n_head)),
        ("attn.k_proj.weight",   "self_attn.k_proj.weight",          ("2d", n_kv_head)),
        ("attn.v_proj.weight",   "self_attn.v_proj.weight",          None),
        ("attn.c_proj.weight",   "self_attn.o_proj.weight",          None),
        ("attn.q_norm.weight",   "self_attn.q_norm.weight",          ("1d",)),
        ("attn.k_norm.weight",   "self_attn.k_norm.weight",          ("1d",)),
        ("mlp.w_gate.weight",    "mlp.gate_proj.weight",             None),
        ("mlp.w_up.weight",      "mlp.up_proj.weight",               None),
        ("mlp.w_down.weight",    "mlp.down_proj.weight",             None),
    ]

    n_per_block_keys_found = 0
    n_permuted = 0
    for i in range(n_unique_layers):
        for src_suffix, dst_suffix, perm_kind in PER_BLOCK_MAP:
            src = f"transformer.h.{i}.{src_suffix}"
            if src not in sd:
                continue
            n_per_block_keys_found += 1
            t = sd[src]
            if perm_kind is not None:
                if perm_kind[0] == "2d":
                    t = _llama_rope_permute_2d(t, perm_kind[1], head_dim)
                elif perm_kind[0] == "1d":
                    t = _llama_rope_permute_1d(t, head_dim)
                n_permuted += 1
            hf_positions = [2 * i, 2 * i + 1] if ws else [i]
            for pos in hf_positions:
                new[f"model.layers.{pos}.{dst_suffix}"] = t.clone()

    logger.info("Remapped %d per-block keys (%d permuted for RoPE) across %d unique layers, ws=%s",
                n_per_block_keys_found, n_permuted, n_unique_layers, ws)
    return new


def build_hf_config(gpt_config: dict, vocab_size: int, modeling_file: str = "modeling_pothana"):
    """Build PothanaConfig dict from our GPTConfig."""
    n_embd = gpt_config["n_embd"]
    n_head = gpt_config["n_head"]
    n_kv_head = gpt_config.get("n_kv_head", n_head)
    n_layer_unique = gpt_config["n_layer"]
    ws = gpt_config.get("use_weight_sharing", False)
    hf_n_layer = n_layer_unique * 2 if ws else n_layer_unique
    head_dim = n_embd // n_head

    # SwiGLU intermediate (matches train_gpt_v2.py)
    inter = int(2 * n_embd * 4 / 3)
    inter = ((inter + 255) // 256) * 256

    return {
        "architectures": ["PothanaForCausalLM"],
        "model_type": "pothana",
        "auto_map": {
            "AutoConfig": f"{modeling_file}.PothanaConfig",
            "AutoModelForCausalLM": f"{modeling_file}.PothanaForCausalLM",
        },
        "torch_dtype": "float32",
        # Architecture
        "hidden_size": n_embd,
        "intermediate_size": inter,
        "num_hidden_layers": hf_n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        # Norm
        "rms_norm_eps": 1e-6,
        # Position
        "max_position_embeddings": gpt_config["block_size"],
        "rope_theta": gpt_config.get("rope_theta", 10000.0),
        "rope_scaling": None,
        # Vocab + tying
        "vocab_size": vocab_size,
        "tie_word_embeddings": gpt_config.get("tie_embeddings", True),
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        # Pothana-specific
        "use_qk_norm": gpt_config.get("use_qk_norm", False),
        # Misc
        "initializer_range": 0.02,
        "use_cache": True,
        "transformers_version": "5.1.0",
    }


def build_generation_config():
    return {
        "_from_model_config": True,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "pad_token_id": 0,
        "transformers_version": "5.1.0",
    }


def convert_tokenizer(tokenizer_dir: Path, output_dir: Path) -> int:
    """Convert tokenizer-new-v5 (custom morfessor_bpe_telugu_v4 format) to HF tokenizers WordLevel.

    Returns the vocab size used by the HF tokenizer (= len(token_to_id)).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing

    with open(tokenizer_dir / "tokenizer.json", "r", encoding="utf-8") as f:
        our_tok = json.load(f)

    assert our_tok.get("type") == "morfessor_bpe_telugu_v4", \
        f"unexpected tokenizer type: {our_tok.get('type')}"

    token_to_id = our_tok["token_to_id"]
    vocab_size = max(token_to_id.values()) + 1
    logger.info("Source tokenizer: %s, vocab=%d", our_tok["type"], vocab_size)

    # HF WordLevel: direct vocab lookup; pre-tokenize on whitespace
    tok = Tokenizer(models.WordLevel(vocab=token_to_id, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = TemplateProcessing(
        single="<bos> $A",
        pair="<bos> $A $B:1",
        special_tokens=[("<bos>", 2)],
    )
    tok.decoder = None  # default space-join; custom PothanaTokenizer handles @@

    # Mark explicit specials
    base_specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    # Plus the 9 retrieval special tokens added at extension time
    retrieval_specials = [
        "<search>", "</search>", "<retrieved>", "</retrieved>",
        "<doc>", "</doc>", "<cite>", "<think>", "</think>",
    ]
    tok.add_special_tokens(base_specials + retrieval_specials)

    out_path = output_dir / "tokenizer.json"
    tok.save(str(out_path))
    logger.info("Saved HF tokenizer.json (%d KB)", out_path.stat().st_size // 1024)

    # tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "auto_map": {
            "AutoTokenizer": [None, "tokenizer_class.PothanaTokenizer"],
        },
        "model_type": "pothana",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 4096,
        "additional_special_tokens": retrieval_specials,
        "extra_info": {
            "type": "morfessor_bpe_telugu_v4",
            "continuation_marker": "@@ (prefix on continuation morphemes)",
            "note": (
                "Inputs are expected to be morfessor-segmented (whitespace-separated morphemes). "
                "For raw Telugu text, run morfessor segmentation first using the included "
                "morfessor_telugu.bin model. The PothanaTokenizer.decode() strips '@@' "
                "continuation markers to reconstruct word forms."
            ),
        },
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    with open(output_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "additional_special_tokens": retrieval_specials,
        }, f, ensure_ascii=False, indent=2)

    return vocab_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--morfessor", type=Path, default=None,
                    help="morfessor_telugu.bin (optional, shipped for raw-text use)")
    ap.add_argument("--modeling-file", type=Path, default=Path("modeling_pothana.py"))
    ap.add_argument("--tokenizer-class-file", type=Path, default=Path("tokenizer_class.py"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    gpt_config = ckpt["config"]
    state_dict = ckpt["model"]
    logger.info("  config: %s", {k: v for k, v in gpt_config.items() if not isinstance(v, (list, dict)) or len(str(v)) < 80})
    logger.info("  step=%s, val_loss=%s, best_val_loss=%s",
                ckpt.get("step"), ckpt.get("val_loss"), ckpt.get("best_val_loss"))
    n_params = sum(t.numel() for t in state_dict.values() if hasattr(t, "numel"))
    logger.info("  source params: %.1fM", n_params / 1e6)

    # 2. Tokenizer (also gives us the vocab_size to use in config)
    logger.info("Converting tokenizer...")
    hf_vocab_size = convert_tokenizer(args.tokenizer, args.output)

    # Sanity: checkpoint vocab must match tokenizer vocab
    ckpt_vocab = gpt_config["vocab_size"]
    if ckpt_vocab != hf_vocab_size:
        logger.warning(
            "vocab_size mismatch: checkpoint=%d, tokenizer=%d. Using tokenizer's %d.",
            ckpt_vocab, hf_vocab_size, hf_vocab_size,
        )

    # 3. Config + generation config
    logger.info("Building config.json...")
    hf_config = build_hf_config(gpt_config, hf_vocab_size,
                                 modeling_file=args.modeling_file.stem)
    with open(args.output / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, ensure_ascii=False, indent=2)
    with open(args.output / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(build_generation_config(), f, ensure_ascii=False, indent=2)

    # 4. Remap and save weights (with RoPE permutation for HF convention)
    logger.info("Remapping state_dict...")
    n_embd = gpt_config["n_embd"]
    n_head = gpt_config["n_head"]
    n_kv_head = gpt_config.get("n_kv_head", n_head)
    head_dim = n_embd // n_head
    new_sd = remap_state_dict(
        state_dict,
        n_unique_layers=gpt_config["n_layer"],
        ws=gpt_config.get("use_weight_sharing", False),
        n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim,
    )
    # Pad embedding/lm_head if vocab mismatch
    if ckpt_vocab != hf_vocab_size and ckpt_vocab < hf_vocab_size:
        for k in ["model.embed_tokens.weight", "lm_head.weight"]:
            if k in new_sd and new_sd[k].size(0) < hf_vocab_size:
                old = new_sd[k]
                pad = torch.zeros(hf_vocab_size - old.size(0), old.size(1), dtype=old.dtype)
                new_sd[k] = torch.cat([old, pad], dim=0)
                logger.info("  padded %s from %d to %d rows", k, old.size(0), hf_vocab_size)
    # Promote to contiguous fp32 for storage
    for k, t in new_sd.items():
        if not t.is_contiguous():
            new_sd[k] = t.contiguous()
    total_params = sum(t.numel() for t in new_sd.values())
    logger.info("  HF state_dict: %d tensors, %.1fM params (after unroll)", len(new_sd), total_params / 1e6)
    logger.info("Saving model.safetensors...")
    save_file(new_sd, str(args.output / "model.safetensors"))
    sz_gb = (args.output / "model.safetensors").stat().st_size / 1e9
    logger.info("  wrote %.2f GB", sz_gb)

    # 5. Copy modeling + tokenizer source files
    logger.info("Copying code files...")
    shutil.copy2(args.modeling_file, args.output / "modeling_pothana.py")
    shutil.copy2(args.tokenizer_class_file, args.output / "tokenizer_class.py")
    if args.morfessor and args.morfessor.exists():
        shutil.copy2(args.morfessor, args.output / "morfessor_telugu.bin")
        logger.info("  copied %s (%.1f MB)", args.morfessor.name, args.morfessor.stat().st_size / 1e6)

    logger.info("\nDone. Output: %s", args.output)
    for p in sorted(args.output.iterdir()):
        logger.info("  %s  (%s)", p.name, _human_size(p.stat().st_size))


def _human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


if __name__ == "__main__":
    main()
