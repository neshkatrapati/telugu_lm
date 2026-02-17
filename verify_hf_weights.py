#!/usr/bin/env python3
"""
Verify HF converted model produces same logits as original checkpoint.

Run this on the REMOTE machine where you have the SFT checkpoint and PyTorch.

Usage:
    # After re-converting with fixed convert_to_hf.py:
    python verify_hf_weights.py --checkpoint ./sft_checkpoints/best.pt --hf-dir ./pothana-chat-300M-hf

    # Or compare against the Hub model:
    python verify_hf_weights.py --checkpoint ./sft_checkpoints/best.pt --hf-dir dvitvaai/pothana-chat-300M

This will:
1. Load the original checkpoint with our GPT model
2. Load the HF model from the specified directory/Hub
3. Feed the same input to both
4. Compare the logits
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--hf-dir", type=str, default="dvitvaai/pothana-chat-300M",
                        help="Path to local HF model dir or Hub model ID")
    parser.add_argument("--tokenizer", "-t", type=Path, default=Path("./tokenizer"))
    args = parser.parse_args()

    hf_model_path = args.hf_dir

    # ---- Load original model ----
    print("=" * 70)
    print("Loading original checkpoint...")
    from train_gpt import GPTConfig, build_model

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)
    model_ours = build_model(config, device="cpu")

    state_dict = checkpoint["model"]
    # Handle torch.compile wrapped models
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v
    model_ours.load_state_dict(cleaned, strict=False)
    model_ours.eval()
    print(f"  Loaded: vocab={config.vocab_size}, layers={config.n_layer}, hidden={config.n_embd}")

    # ---- Load HF model ----
    print(f"\nLoading HF model from {hf_model_path}...")
    from transformers import AutoModelForCausalLM
    model_hf = AutoModelForCausalLM.from_pretrained(
        hf_model_path, torch_dtype=torch.float32
    ).eval()
    print(f"  Loaded: vocab={model_hf.config.vocab_size}")

    # ---- Test input ----
    # Use the exact chat prompt IDs
    test_ids = [2, 86071, 72, 26, 1605, 4808, 260, 67268, 12485, 66578,
                86074, 86072, 172, 1421, 12867, 66590, 86074, 86073]
    input_tensor = torch.tensor([test_ids], dtype=torch.long)
    print(f"\nTest input: {len(test_ids)} tokens")

    # ---- Forward pass: original model ----
    print("\nOriginal model forward pass...")
    with torch.no_grad():
        logits_ours = model_ours(input_tensor)[0]
        print(f"  Our logits shape: {logits_ours.shape}")

    # ---- Forward pass: HF model ----
    print("HF model forward pass...")
    with torch.no_grad():
        output_hf = model_hf(input_tensor)
        logits_hf = output_hf.logits
        print(f"  HF logits shape: {logits_hf.shape}")

    # ---- Compare logits ----
    print("\n" + "=" * 70)
    print("COMPARING LOGITS")
    print("=" * 70)

    # Our model returns (B, 1, V) for the last position when targets=None
    # HF returns (B, T, V) for all positions
    # Compare the last position
    if logits_ours.shape[1] == 1:
        logits_ours_last = logits_ours[:, 0, :]  # (B, V)
    else:
        logits_ours_last = logits_ours[:, -1, :]  # (B, V)

    logits_hf_last = logits_hf[:, -1, :]  # (B, V)

    # Truncate to same vocab size (ours might be smaller)
    min_v = min(logits_ours_last.shape[-1], logits_hf_last.shape[-1])
    lo = logits_ours_last[:, :min_v]
    lh = logits_hf_last[:, :min_v]

    diff = (lo - lh).abs()
    print(f"\n  Compared vocab range: [0, {min_v})")
    print(f"  Max absolute diff:  {diff.max().item():.6f}")
    print(f"  Mean absolute diff: {diff.mean().item():.6f}")
    print(f"  Relative diff:      {(diff / (lo.abs() + 1e-8)).mean().item():.6f}")

    # Check top predicted tokens
    topk_ours = lo.topk(10, dim=-1)
    topk_hf = lh.topk(10, dim=-1)

    print(f"\n  Top-10 tokens (original): {topk_ours.indices[0].tolist()}")
    print(f"  Top-10 tokens (HF):       {topk_hf.indices[0].tolist()}")
    print(f"  Top-10 logits (original): {[f'{v:.2f}' for v in topk_ours.values[0].tolist()]}")
    print(f"  Top-10 logits (HF):       {[f'{v:.2f}' for v in topk_hf.values[0].tolist()]}")

    if diff.max().item() < 0.01:
        print("\n  ✅ MATCH — logits are essentially identical")
    elif diff.max().item() < 1.0:
        print("\n  ⚠️  CLOSE — small numerical differences (likely float precision)")
    else:
        print("\n  ❌ MISMATCH — logits are significantly different!")
        print("     The weight conversion has a bug.")

        # Try to identify where the mismatch starts
        print("\n  Debugging: comparing intermediate outputs...")

        with torch.no_grad():
            # Our embedding
            embed_ours = model_ours.transformer.wte(input_tensor)
            # HF embedding
            embed_hf = model_hf.model.embed_tokens(input_tensor)

            embed_diff = (embed_ours - embed_hf).abs().max().item()
            print(f"    Embedding diff: {embed_diff:.6f}")

            if embed_diff > 0.01:
                print("    → Embedding weights mismatch!")
            else:
                print("    → Embeddings match. Issue is in transformer layers.")

                # Check layer 0
                x_ours = model_ours.transformer.drop(embed_ours)
                freqs = model_ours.freqs_cis[:input_tensor.shape[1]]
                freqs_cis = torch.view_as_complex(freqs)

                # Layer 0 attention input
                ln1_ours = model_ours.transformer.h[0].ln_1(x_ours)
                ln1_hf = model_hf.model.layers[0].input_layernorm(embed_hf)
                ln1_diff = (ln1_ours - ln1_hf).abs().max().item()
                print(f"    Layer 0 LN1 diff: {ln1_diff:.6f}")

                # QKV raw (before RoPE)
                attn_ours = model_ours.transformer.h[0].attn
                qkv_ours = attn_ours.c_attn(ln1_ours)
                q_ours, k_ours, v_ours = qkv_ours.split(config.n_embd, dim=2)

                attn_hf = model_hf.model.layers[0].self_attn
                q_hf = attn_hf.q_proj(ln1_hf)
                k_hf = attn_hf.k_proj(ln1_hf)
                v_hf = attn_hf.v_proj(ln1_hf)

                # V should match exactly (no RoPE permutation)
                v_diff = (v_ours - v_hf).abs().max().item()
                print(f"    Layer 0 V diff: {v_diff:.6f}")

                # Q and K won't match raw (due to RoPE permutation),
                # but they should match after applying their respective RoPE
                B, T, C = ln1_ours.shape
                H = config.n_head
                D = C // H

                # Our RoPE: view_as_complex on consecutive pairs
                q_ours_r = q_ours.view(B, T, H, D).transpose(1, 2)
                k_ours_r = k_ours.view(B, T, H, D).transpose(1, 2)
                from train_gpt import apply_rotary_emb
                q_ours_rope, k_ours_rope = apply_rotary_emb(q_ours_r, k_ours_r, freqs_cis)

                # HF RoPE: rotate_half on half-split
                q_hf_r = q_hf.view(B, T, H, D).transpose(1, 2)
                k_hf_r = k_hf.view(B, T, H, D).transpose(1, 2)
                cos_sin = model_hf.model.layers[0].self_attn.rotary_emb(q_hf_r, torch.arange(T).unsqueeze(0))
                cos, sin = cos_sin
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q_hf_rope, k_hf_rope = apply_rotary_pos_emb(q_hf_r, k_hf_r, cos, sin)

                q_rope_diff = (q_ours_rope - q_hf_rope).abs().max().item()
                k_rope_diff = (k_ours_rope - k_hf_rope).abs().max().item()
                print(f"    Layer 0 Q (after RoPE) diff: {q_rope_diff:.6f}")
                print(f"    Layer 0 K (after RoPE) diff: {k_rope_diff:.6f}")

                if q_rope_diff < 0.001 and k_rope_diff < 0.001:
                    print("    → Q/K match after RoPE! RoPE permutation is correct.")
                else:
                    print("    → Q/K still differ after RoPE. Check permutation logic.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
