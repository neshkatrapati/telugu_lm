#!/usr/bin/env python3
"""
Debug Learned Engrams v2 — inspect what the module is actually doing.

Loads a checkpoint, runs a sample through the model, and captures
intermediate engram values: mask probabilities, gate activations,
table embedding norms, hash slot usage, and output contributions.

Usage:
    python debug_engrams.py --checkpoint ./checkpoints/best.pt \
        --tokenizer ./tokenizer --prompt "తెలుగు భాష చాలా"
"""

import argparse
import sys
import math
import torch
import numpy as np
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Debug engram v2 module internals")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer dir")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (optional)")
    parser.add_argument("--use-val-data", type=str, default=None,
                        help="Path to val.bin — use a random chunk instead of prompt")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for val data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load checkpoint ---
    from train_gpt import GPTConfig, build_model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    config = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg.get("n_kv_head", cfg["n_head"]),
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
        bias=cfg["bias"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        use_weight_sharing=cfg.get("use_weight_sharing", False),
        use_engrams=cfg.get("use_engrams", False),
        engram_table_size=cfg.get("engram_table_size", 100_000),
        engram_dim=cfg.get("engram_dim", 128),
        engram_window=cfg.get("engram_window", 6),
        engram_inject_indices=tuple(cfg.get("engram_inject_indices", (4, 30))),
    )

    if not config.use_engrams:
        print("ERROR: This checkpoint was trained without engrams (use_engrams=False)")
        sys.exit(1)

    print(f"Model: {config.n_layer}L, {config.n_embd}d, engrams at indices {config.engram_inject_indices}")
    print(f"Engram v2 config: table={config.engram_table_size}, dim={config.engram_dim}, "
          f"window={config.engram_window}")

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    if hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()
    model.to(device)
    if hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()
    model.eval()

    step = checkpoint.get("step", "?")
    print(f"Loaded checkpoint at step {step}")
    print()

    # --- Get input tokens ---
    if args.use_val_data:
        data = np.memmap(args.use_val_data, dtype=np.uint32, mode="r")
        start = np.random.randint(0, len(data) - args.seq_len - 1)
        token_ids = data[start:start + args.seq_len].astype(np.int64).tolist()
        print(f"Using val data chunk: positions [{start}:{start + args.seq_len}]")
    elif args.prompt:
        # Tokenize the prompt
        tokenizer_dir = Path(args.tokenizer)
        sp_model_path = tokenizer_dir / "sp_telugu.model"
        if sp_model_path.exists():
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
            token_ids = [sp.bos_id()] + sp.encode(args.prompt, out_type=int)
        else:
            sys.path.insert(0, str(Path(__file__).parent))
            from train_tokenizer import MorfessorTokenizer
            tok = MorfessorTokenizer(tokenizer_dir)
            token_ids = tok.encode(args.prompt, add_bos=True, add_eos=False)
        print(f"Prompt: {args.prompt}")
    else:
        # Random tokens
        token_ids = list(range(10, 10 + args.seq_len))
        print(f"Using sequential token IDs (no prompt or val data provided)")

    print(f"Input length: {len(token_ids)} tokens")
    print()

    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    # === Manual instrumented forward pass ===
    captures = {}

    print("Running instrumented forward pass...")
    print()

    with torch.no_grad():
        B, T = idx.size()
        tok_emb = model.transformer.wte(idx)
        x = model.transformer.drop(tok_emb)
        freqs_cis = torch.view_as_complex(model.freqs_cis[:T])

        active_map = getattr(model, "_engram_map_active", model._engram_map)

        for sched_idx, block in enumerate(model._block_schedule):
            # Run block normally (v2: engram is AFTER block)
            x = block(x, freqs_cis)

            if sched_idx in active_map:
                mod_idx = active_map[sched_idx]
                engram_mod = model.engram_modules[mod_idx]
                W = engram_mod.W

                # --- Manually run engram internals for diagnostics ---
                # 1. Build window IDs
                sentinel_pad = torch.full(
                    (B, W), engram_mod.sentinel_val, device=device, dtype=idx.dtype
                )
                padded_ids = torch.cat([sentinel_pad, idx], dim=1)
                window_ids = padded_ids.unfold(1, W, 1)[:, :T, :]

                # 2. Mask predictor
                mask_logits = engram_mod.mask_predictor(x)
                p_w = torch.sigmoid(mask_logits)
                hard_mask = (p_w > 0.5).float()

                # 3. Hash keys
                sentinel_fill = torch.full_like(window_ids, engram_mod.sentinel_val)
                mask_int = hard_mask.long()
                keys = window_ids * mask_int + sentinel_fill * (1 - mask_int)

                # 4. Hash → indices
                hash_vals = (keys.float() * engram_mod.hash_weights.float()).sum(dim=-1).long()
                indices = hash_vals.abs() % config.engram_table_size

                # 5. Run full engram forward for output
                engram_out, slot_idx = engram_mod(
                    x, idx, model.memory_table, config.engram_table_size
                )

                # 6. Gate values
                gate = torch.sigmoid(
                    engram_mod.gate_proj(engram_mod.ln_gate(x))
                )

                # --- Capture everything ---
                captures[f"engram_{mod_idx}"] = {
                    "mask_probs": p_w.detach().cpu(),
                    "hard_mask": hard_mask.detach().cpu(),
                    "mask_logits": mask_logits.detach().cpu(),
                    "output_norm": engram_out.detach().norm(dim=-1).cpu(),
                    "hidden_norm": x.detach().norm(dim=-1).cpu(),
                    "gate_values": gate.detach().cpu(),
                    "positions_active": hard_mask.sum(dim=-1).detach().cpu(),
                    "slot_indices": indices.detach().cpu(),
                }

                x = x + engram_out

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), idx.view(-1), ignore_index=-1)

    print(f"Forward pass loss: {loss.item():.4f}")
    print()

    # === Analyze captures ===
    for name, cap in captures.items():
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        p_w = cap["mask_probs"]       # (1, T, W)
        hard = cap["hard_mask"]       # (1, T, W)
        out_norm = cap["output_norm"] # (1, T)
        h_norm = cap["hidden_norm"]   # (1, T)
        gate = cap["gate_values"]     # (1, T, 1)
        positions_active = cap["positions_active"]  # (1, T)
        slot_idx = cap["slot_indices"]  # (1, T)

        T_seq = p_w.size(1)
        W = p_w.size(2)

        # 1. Mask probability statistics
        print(f"\n--- Mask Probabilities (should vary if learning) ---")
        print(f"  Mean p(mask=1):  {p_w.mean().item():.4f}")
        print(f"  Std p(mask=1):   {p_w.std().item():.4f}  (>0.1 = good variation)")
        print(f"  Min:             {p_w.min().item():.4f}")
        print(f"  Max:             {p_w.max().item():.4f}")

        # Per-window-position mean
        per_pos = p_w[0].mean(dim=0)  # (W,)
        print(f"  Per window pos:  {[f'{v:.3f}' for v in per_pos.tolist()]}")

        # 2. Hard mask statistics
        print(f"\n--- Hard Mask (positions selected per query) ---")
        mean_active = positions_active.float().mean().item()
        print(f"  Mean active:     {mean_active:.2f} / {W}")
        # Distribution of active counts
        counts = positions_active[0].long().tolist()
        dist = Counter(counts)
        print(f"  Distribution:    {dict(sorted(dist.items()))}")

        # 3. Are all masks identical? (collapsed)
        mask_patterns = hard[0]  # (T, W)
        unique_patterns = torch.unique(mask_patterns, dim=0)
        print(f"  Unique patterns: {len(unique_patterns)} / {T_seq} positions  ", end="")
        if len(unique_patterns) <= 3:
            print("*** COLLAPSED — all positions use same mask ***")
            for i, pat in enumerate(unique_patterns):
                print(f"    Pattern {i}: {pat.long().tolist()}")
        else:
            print("(good — diverse patterns)")

        # 4. Gate statistics
        print(f"\n--- Gate Values (how much engram output is used) ---")
        print(f"  Mean gate:  {gate.mean().item():.4f}")
        print(f"  Std gate:   {gate.std().item():.4f}")
        print(f"  Min:        {gate.min().item():.4f}")
        print(f"  Max:        {gate.max().item():.4f}")

        # 5. Output contribution
        ratio = out_norm / (h_norm + 1e-8)
        print(f"\n--- Engram Output vs Hidden State ---")
        print(f"  Hidden state norm:  mean={h_norm.mean().item():.2f}")
        print(f"  Engram output norm: mean={out_norm.mean().item():.6f}")
        print(f"  Ratio (out/hidden): mean={ratio.mean().item():.6f}  ", end="")
        if ratio.mean().item() < 0.001:
            print("*** NEGLIGIBLE — engrams contributing almost nothing ***")
        elif ratio.mean().item() < 0.01:
            print("(small but nonzero)")
        else:
            print("(meaningful contribution)")

        # 6. Slot collision analysis
        print(f"\n--- Slot Usage ---")
        flat_slots = slot_idx[0].tolist()
        unique_slots = len(set(flat_slots))
        print(f"  Unique slots accessed: {unique_slots} / {T_seq} positions")
        print(f"  Collision rate:        {1.0 - unique_slots / T_seq:.2%}")

        print()

    # === Memory table statistics ===
    print(f"{'='*60}")
    print(f"  Memory Table Statistics")
    print(f"{'='*60}")
    table_weight = model.memory_table.weight.data  # (table_size, engram_dim)
    row_norms = table_weight.norm(dim=1)  # (table_size,)
    nonzero_rows = (row_norms > 1e-6).sum().item()
    print(f"  Table size:      {table_weight.size(0):,} x {table_weight.size(1)}")
    print(f"  Non-zero rows:   {nonzero_rows:,} / {table_weight.size(0):,}  "
          f"({100*nonzero_rows/table_weight.size(0):.2f}%)")
    print(f"  Row norm — mean: {row_norms.mean().item():.6f}")
    print(f"  Row norm — max:  {row_norms.max().item():.6f}")
    print(f"  Row norm — std:  {row_norms.std().item():.6f}")
    if nonzero_rows > 0:
        nz_norms = row_norms[row_norms > 1e-6]
        print(f"  Non-zero norms — mean: {nz_norms.mean().item():.6f}, max: {nz_norms.max().item():.6f}")

    # Top-K most used slots (by norm, as proxy for training activity)
    if nonzero_rows > 0:
        top_k = min(10, nonzero_rows)
        top_norms, top_indices = row_norms.topk(top_k)
        print(f"\n  Top {top_k} slots by norm:")
        for i in range(top_k):
            print(f"    Slot {top_indices[i].item():6d}: norm={top_norms[i].item():.6f}")

    # Weight magnitude of key module parameters
    print(f"\n--- Module Parameter Norms ---")
    for i, em in enumerate(model.engram_modules):
        print(f"  Engram {i}:")
        print(f"    mask_predictor[0].weight norm: {em.mask_predictor[0].weight.data.norm().item():.4f}")
        print(f"    mask_predictor[2].weight norm: {em.mask_predictor[2].weight.data.norm().item():.4f}")
        print(f"    W_V.weight norm:               {em.W_V.weight.data.norm().item():.6f}  (zero-init)")
        print(f"    gate_proj.weight norm:         {em.gate_proj.weight.data.norm().item():.6f}  (zero-init)")
        print(f"    gate_proj.bias:                {em.gate_proj.bias.data.item():.6f}  (zero-init)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
