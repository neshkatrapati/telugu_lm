#!/usr/bin/env python3
"""
Debug Learned Engrams — inspect what the module is actually doing.

Loads a checkpoint, runs a sample through the model, and captures
intermediate engram values: mask probabilities, gate activations,
table embedding norms, hash collisions, and output contributions.

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
    parser = argparse.ArgumentParser(description="Debug engram module internals")
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
        engram_table_size=cfg.get("engram_table_size", 1_000_000),
        engram_dim=cfg.get("engram_dim", 128),
        engram_n_heads=cfg.get("engram_n_heads", 4),
        engram_window=cfg.get("engram_window", 6),
        engram_inject_indices=tuple(cfg.get("engram_inject_indices", (4, 30))),
    )

    if not config.use_engrams:
        print("ERROR: This checkpoint was trained without engrams (use_engrams=False)")
        sys.exit(1)

    print(f"Model: {config.n_layer}L, {config.n_embd}d, engrams at indices {config.engram_inject_indices}")
    print(f"Engram config: table={config.engram_table_size}, dim={config.engram_dim}, "
          f"heads={config.engram_n_heads}, window={config.engram_window}")

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    if hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()
    model.to(device)
    if hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()
    model.eval()
    model._current_tau = 0.3  # inference tau

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

    # === Capture engram internals by instrumenting the pattern_mlp ===
    # We use register_forward_hook on sub-modules + manual inspection
    # instead of monkey-patching forward (avoids self/binding issues)
    captures = {}

    def capture_engram_state(module_idx, engram_mod):
        """Manually run engram diagnostics on the module's internal state."""
        cap = {}

        # Inspect pattern_mlp weights
        mlp = engram_mod.pattern_mlp
        cap["mlp0_weight_norm"] = mlp[0].weight.data.norm().item()
        cap["mlp2_weight_norm"] = mlp[2].weight.data.norm().item()
        cap["wk_norm"] = engram_mod.W_K.weight.data.norm().item()
        cap["wv_norm"] = engram_mod.W_V.weight.data.norm().item()
        cap["conv_norm"] = engram_mod.conv.weight.data.norm().item()

        return cap

    # Pre-capture module parameter norms
    for i, em in enumerate(model.engram_modules):
        captures[f"params_{i}"] = capture_engram_state(i, em)

    # Now do a manual instrumented forward pass through the engram blocks
    # Instead of hooking, we'll run the model and intercept at the Block level
    print("Running instrumented forward pass...")
    print()

    with torch.no_grad():
        B, T = idx.size()
        tok_emb = model.transformer.wte(idx)
        x = model.transformer.drop(tok_emb)
        freqs_cis = torch.view_as_complex(model.freqs_cis[:T])

        active_map = getattr(model, "_engram_map_active", model._engram_map)
        tau = getattr(model, "_current_tau", 0.3)

        for sched_idx, block in enumerate(model._block_schedule):
            if sched_idx in active_map:
                mod_idx = active_map[sched_idx]
                engram_mod = model.engram_modules[mod_idx]
                W = engram_mod.W

                # Run attention with QK logits
                attn_out, qk_logits = block.attn(block.ln_1(x), freqs_cis, return_qk=True)
                h = x + attn_out

                # --- Extract QK window (same as engram forward) ---
                offsets = torch.arange(W, device=device).unsqueeze(0)
                positions = torch.arange(T, device=device).unsqueeze(1)
                key_indices = positions - W + offsets
                valid_mask = key_indices >= 0
                key_indices_clamped = key_indices.clamp(min=0)
                ki_expanded = key_indices_clamped.unsqueeze(0).unsqueeze(0).expand(
                    B, qk_logits.size(1), -1, -1)
                qk_window = torch.gather(qk_logits, dim=3, index=ki_expanded)
                valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand_as(qk_window)
                qk_window = qk_window.masked_fill(~valid_mask_expanded, 0.0)
                qk_window = qk_window.mean(dim=1)  # (B, T, W)

                # --- Pattern predictor ---
                mask_logits = engram_mod.pattern_mlp(qk_window)
                p_w = torch.sigmoid(mask_logits)
                hard_mask = (p_w > 0.5).float()

                # --- Run full engram forward for output ---
                engram_out, mdl_loss = engram_mod(
                    h, qk_logits, idx, model.memory_table, tau
                )

                # --- Capture everything ---
                captures[f"engram_{mod_idx}"] = {
                    "mask_probs": p_w.detach().cpu(),
                    "hard_mask": hard_mask.detach().cpu(),
                    "mask_logits": mask_logits.detach().cpu(),
                    "qk_window": qk_window.detach().cpu(),
                    "output_norm": engram_out.detach().norm(dim=-1).cpu(),
                    "hidden_norm": h.detach().norm(dim=-1).cpu(),
                    "mdl_loss": mdl_loss.item(),
                    "positions_active": hard_mask.sum(dim=-1).detach().cpu(),
                }

                x = h + engram_out
                x = x + block.mlp(block.ln_2(x))
            else:
                x = block(x, freqs_cis)

        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), idx.view(-1), ignore_index=-1)

    print(f"Forward pass loss: {loss.item():.4f}")
    print()

    # === Analyze captures ===
    for name, cap in captures.items():
        if name.startswith("params_"):
            continue  # handled separately below
        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        p_w = cap["mask_probs"]       # (1, T, W)
        hard = cap["hard_mask"]       # (1, T, W)
        out_norm = cap["output_norm"] # (1, T)
        h_norm = cap["hidden_norm"]   # (1, T)
        positions_active = cap["positions_active"]  # (1, T)
        qk_win = cap["qk_window"]    # (1, T, W)

        T = p_w.size(1)
        W = p_w.size(2)

        # 1. Mask probability statistics
        print(f"\n--- Mask Probabilities (should vary if learning) ---")
        print(f"  Mean p(mask=1):  {p_w.mean().item():.4f}  (prior={config.engram_mdl_prior})")
        print(f"  Std p(mask=1):   {p_w.std().item():.4f}  (>0.1 = good variation)")
        print(f"  Min:             {p_w.min().item():.4f}")
        print(f"  Max:             {p_w.max().item():.4f}")

        # Per-window-position mean
        per_pos = p_w[0].mean(dim=0)  # (W,)
        print(f"  Per window pos:  {[f'{v:.3f}' for v in per_pos.tolist()]}")

        # 2. Hard mask statistics
        print(f"\n--- Hard Mask (positions selected per query) ---")
        mean_active = positions_active.float().mean().item()
        print(f"  Mean active:     {mean_active:.2f} / {W}  (expect ~{W*config.engram_mdl_prior:.1f} from prior)")
        # Distribution of active counts
        counts = positions_active[0].long().tolist()
        dist = Counter(counts)
        print(f"  Distribution:    {dict(sorted(dist.items()))}")

        # 3. Are all masks identical? (collapsed)
        mask_patterns = hard[0]  # (T, W)
        unique_patterns = torch.unique(mask_patterns, dim=0)
        print(f"  Unique patterns: {len(unique_patterns)} / {T} positions  ", end="")
        if len(unique_patterns) <= 3:
            print("*** COLLAPSED — all positions use same mask ***")
            for i, pat in enumerate(unique_patterns):
                print(f"    Pattern {i}: {pat.long().tolist()}")
        else:
            print("(good — diverse patterns)")

        # 4. QK window logit statistics
        print(f"\n--- QK Window Logits (input to pattern predictor) ---")
        print(f"  Mean:  {qk_win.mean().item():.4f}")
        print(f"  Std:   {qk_win.std().item():.4f}  (>0.1 = varied attention)")
        print(f"  Range: [{qk_win.min().item():.4f}, {qk_win.max().item():.4f}]")

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

        # 6. MDL loss
        print(f"\n--- MDL Loss ---")
        print(f"  Value: {cap['mdl_loss']:.8f}")

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

    # Weight magnitude of key module parameters
    print(f"\n--- Module Parameter Norms ---")
    for i, em in enumerate(model.engram_modules):
        print(f"  Engram {i}:")
        print(f"    pattern_mlp[0].weight norm: {em.pattern_mlp[0].weight.data.norm().item():.4f}")
        print(f"    pattern_mlp[2].weight norm: {em.pattern_mlp[2].weight.data.norm().item():.4f}")
        print(f"    W_K.weight norm:            {em.W_K.weight.data.norm().item():.4f}")
        print(f"    W_V.weight norm:            {em.W_V.weight.data.norm().item():.6f}  (zero-init)")
        print(f"    conv.weight norm:           {em.conv.weight.data.norm().item():.6f}  (zero-init)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
