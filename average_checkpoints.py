#!/usr/bin/env python3
"""
Checkpoint averaging (SWA / EMA) + validation eval.

Usage (typical, run after Base v2 finishes):
    python average_checkpoints.py \
        --ckpt-dir checkpoints/base-v2 \
        --start-step 6000 --end-step 8000 \
        --data train-data-v2 \
        --tokenizer tokenizer-new-v5 \
        --output-dir checkpoints/base-v2-averaged

Produces:
  checkpoints/base-v2-averaged/swa.pt          - equal-weight average
  checkpoints/base-v2-averaged/ema_linear.pt   - linearly increasing weights
  checkpoints/base-v2-averaged/ema_geom.pt     - geometric decay weights (recent-heavy)

And evaluates each, plus original best.pt and final.pt for comparison.
Reports val loss to stdout AND writes a results.json.

CPU-only averaging uses minimal RAM (incremental accumulator, one ckpt at a time).
Evaluation uses GPU if available, falls back to CPU.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def discover_checkpoints(ckpt_dir: Path, start_step: int, end_step: int):
    """Find step_NNNNN.pt files in range [start_step, end_step]."""
    pattern = re.compile(r"step_0*(\d+)\.pt$")
    found = []
    for p in sorted(ckpt_dir.glob("step_*.pt")):
        m = pattern.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if start_step <= step <= end_step:
            found.append((step, p))
    found.sort(key=lambda x: x[0])
    return found


def load_state_dict_only(path: Path):
    """Load just the model state_dict from a checkpoint. Returns (sd, meta)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if "model" not in ckpt:
        raise SystemExit(f"checkpoint {path} has no 'model' key")
    sd = ckpt["model"]
    meta = {k: ckpt[k] for k in ckpt if k != "model" and k != "optimizer" and k != "table_optimizer"}
    return sd, meta


def compute_average(paths, weights, label):
    """Streaming weighted average of model state_dicts.

    Args:
        paths: list of Path
        weights: list of floats, will be normalized to sum to 1
        label: str for logging
    Returns:
        averaged_state_dict, template_meta (from last checkpoint)
    """
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    print(f"\n[{label}] averaging {len(paths)} checkpoints", flush=True)
    for p, wi in zip(paths, w):
        print(f"    weight={wi:.4f}  {p.name}", flush=True)

    # Initialize accumulator from first checkpoint
    avg = None
    keys = None
    dtypes = None

    t0 = time.time()
    for i, (path, wi) in enumerate(zip(paths, w)):
        print(f"  [{i+1}/{len(paths)}] loading {path.name}...", flush=True)
        sd, meta = load_state_dict_only(path)
        if avg is None:
            keys = list(sd.keys())
            dtypes = {k: sd[k].dtype for k in keys}
            avg = {k: torch.zeros_like(sd[k], dtype=torch.float32) for k in keys}
        for k in keys:
            t = sd[k]
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            avg[k].add_(t, alpha=float(wi))
        del sd

    # Convert back to original dtypes
    for k in keys:
        if avg[k].dtype != dtypes[k]:
            avg[k] = avg[k].to(dtypes[k])

    # Get metadata template from the LAST (most recent) checkpoint
    _, template_meta = load_state_dict_only(paths[-1])
    print(f"  [{label}] done in {time.time()-t0:.1f}s", flush=True)
    return avg, template_meta


def save_avg_checkpoint(avg_sd, template_meta, source_paths, weights, label, output_path):
    """Save the averaged state_dict as a checkpoint file."""
    out = dict(template_meta)
    out["model"] = avg_sd
    out["averaging"] = {
        "strategy": label,
        "source_paths": [str(p) for p in source_paths],
        "weights": [float(w) for w in weights],
        "n_checkpoints": len(source_paths),
    }
    # Remove optimizer-related metadata if any leaked through
    for k in ["optimizer", "table_optimizer"]:
        out.pop(k, None)
    torch.save(out, str(output_path))
    sz_mb = os.path.getsize(output_path) / 1e6
    print(f"  saved {output_path} ({sz_mb:.1f} MB)", flush=True)


def build_model_from_trainer(train_gpt_v2_path: Path, model_config):
    """Import build_model from train_gpt_v2.py."""
    spec = importlib.util.spec_from_file_location("tgv2", str(train_gpt_v2_path))
    tgv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgv2)
    return tgv2.build_model(model_config), tgv2


def evaluate_checkpoint(ckpt_path: Path, train_gpt_v2_path: Path, data_path: Path,
                       n_batches: int, batch_size: int, block_size: int,
                       device: str, seed: int = 1234):
    """Load a checkpoint's model weights into a fresh model, evaluate on val.bin.

    Returns dict: {val_loss, n_batches, tokens_evaluated}
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    cfg_dict = ckpt["config"] if isinstance(ckpt.get("config"), dict) else None
    if cfg_dict is None:
        raise SystemExit(f"{ckpt_path}: no config dict in checkpoint")

    # Reconstruct GPTConfig from the dict
    spec = importlib.util.spec_from_file_location("tgv2", str(train_gpt_v2_path))
    tgv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgv2)
    GPTConfig = tgv2.GPTConfig

    # Build config — start with defaults, override from saved config
    config_kwargs = {}
    for field in GPTConfig.__dataclass_fields__:
        if field in cfg_dict:
            config_kwargs[field] = cfg_dict[field]
    model_cfg = GPTConfig(**config_kwargs)

    # Build model and load weights
    model = tgv2.build_model(model_cfg, device=device)
    model = model.to(device)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # OK if it's just freqs_cis (recomputed buffer)
        unexpected_extra = [k for k in missing if "freqs_cis" not in k]
        if unexpected_extra:
            print(f"    WARN: missing keys (non-freqs_cis): {unexpected_extra[:5]}", flush=True)
    if unexpected:
        print(f"    WARN: unexpected keys: {unexpected[:5]}", flush=True)

    # Free original ckpt
    del ckpt
    del sd

    model.eval()
    # Data
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    n_tok = len(data)
    rng = np.random.RandomState(seed)

    losses = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_batches):
            ix = rng.randint(0, n_tok - block_size - 1, (batch_size,))
            x = np.stack([data[i:i+block_size].astype(np.int64) for i in ix])
            y = np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix])
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            # Use autocast for speed
            with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                     dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
                logits, loss = model(x, targets=y)
            losses.append(loss.item())
    elapsed = time.time() - t0
    val_loss = float(np.mean(losses))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "val_loss": val_loss,
        "val_loss_std": float(np.std(losses)),
        "n_batches": n_batches,
        "tokens_evaluated": int(n_batches * batch_size * block_size),
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="Directory containing step_NNNNN.pt files (also best.pt, final.pt)")
    ap.add_argument("--start-step", type=int, default=6000,
                    help="First checkpoint step to include (default: 6000 — after WSD decay starts)")
    ap.add_argument("--end-step", type=int, default=10**9,
                    help="Last checkpoint step (default: unlimited)")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write averaged checkpoints (default: <ckpt-dir>-averaged)")
    ap.add_argument("--train-gpt-py", type=Path, default=Path("train_gpt_v2.py"),
                    help="Path to train_gpt_v2.py (for build_model + GPTConfig)")
    ap.add_argument("--data", type=Path, default=None,
                    help="Path to data dir with val.bin (e.g. train-data-v2). Skip eval if not given.")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="Tokenizer dir (not used directly, but recorded)")
    ap.add_argument("--eval-batches", type=int, default=40,
                    help="Validation batches per checkpoint (default: 40)")
    ap.add_argument("--eval-batch-size", type=int, default=8,
                    help="Batch size during eval (default: 8 — keep small for memory)")
    ap.add_argument("--eval-block-size", type=int, default=4096)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--device", type=str, default=None,
                    help="'cuda' / 'cpu' / 'cuda:0' etc. Auto if not given.")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Just compute averages, don't evaluate")
    args = ap.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        args.output_dir = args.ckpt_dir.parent / (args.ckpt_dir.name + "-averaged")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={args.device}")
    if args.device.startswith("cuda"):
        print(f"        gpu: {torch.cuda.get_device_name(0)}")

    # Discover
    print(f"\n[discover] {args.ckpt_dir}  range=[{args.start_step}, {args.end_step}]")
    found = discover_checkpoints(args.ckpt_dir, args.start_step, args.end_step)
    if not found:
        raise SystemExit(f"no step_NNNNN.pt checkpoints found in {args.ckpt_dir} in range")
    for step, p in found:
        print(f"    step={step:>5}  {p.name}  ({os.path.getsize(p)/1e9:.2f} GB)")
    steps = [s for s, _ in found]
    paths = [p for _, p in found]

    # --- Strategy 1: SWA (equal weights) ---
    weights_swa = [1.0] * len(paths)
    avg_swa, meta = compute_average(paths, weights_swa, "swa-equal")
    swa_path = args.output_dir / "swa.pt"
    save_avg_checkpoint(avg_swa, meta, paths, weights_swa, "swa-equal", swa_path)
    del avg_swa

    # --- Strategy 2: EMA-linear (weight ∝ step number) ---
    # Recent checkpoints weighted more, linearly
    weights_lin = [float(s) for s in steps]
    avg_lin, meta = compute_average(paths, weights_lin, "ema-linear")
    lin_path = args.output_dir / "ema_linear.pt"
    save_avg_checkpoint(avg_lin, meta, paths, weights_lin, "ema-linear", lin_path)
    del avg_lin

    # --- Strategy 3: EMA-geometric (decay-based) ---
    # The most recent gets weight 1, prev gets decay, prev gets decay^2, etc.
    decay = 0.9
    n = len(paths)
    weights_geom = [decay ** (n - 1 - i) for i in range(n)]
    avg_geom, meta = compute_average(paths, weights_geom, "ema-geom")
    geom_path = args.output_dir / "ema_geom.pt"
    save_avg_checkpoint(avg_geom, meta, paths, weights_geom, "ema-geom", geom_path)
    del avg_geom

    if args.skip_eval or args.data is None:
        print("\n[done] averaging only (no eval requested)")
        return

    # --- Evaluation ---
    val_bin = args.data / "val.bin"
    if not val_bin.exists():
        print(f"WARN: {val_bin} not found, skipping eval")
        return

    eval_candidates = []
    # Original anchors for comparison
    for nm in ["final.pt", "best.pt"]:
        p = args.ckpt_dir / nm
        if p.exists():
            eval_candidates.append(("original/" + nm.replace(".pt",""), p))
    # Last step checkpoint
    eval_candidates.append((f"original/step_{steps[-1]}", paths[-1]))
    # Our averages
    eval_candidates.append(("swa-equal", swa_path))
    eval_candidates.append(("ema-linear", lin_path))
    eval_candidates.append(("ema-geom-d09", geom_path))

    print(f"\n[eval] running {args.eval_batches} batches of {args.eval_batch_size}x{args.eval_block_size} tokens each")
    print(f"       on {val_bin} (deterministic, seed={args.eval_seed})")

    results = {}
    for label, p in eval_candidates:
        print(f"\n  -- evaluating {label}: {p.name} --", flush=True)
        try:
            r = evaluate_checkpoint(
                p, args.train_gpt_py, val_bin,
                n_batches=args.eval_batches,
                batch_size=args.eval_batch_size,
                block_size=args.eval_block_size,
                device=args.device,
                seed=args.eval_seed,
            )
            print(f"     val_loss={r['val_loss']:.4f}  (std {r['val_loss_std']:.4f}, {r['elapsed_sec']:.1f}s)")
            results[label] = r
        except Exception as e:
            print(f"     FAILED: {type(e).__name__}: {e}")
            results[label] = {"error": str(e)}

    # Write results.json + print summary table
    out_json = args.output_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump({
            "discovered_steps": steps,
            "eval_config": {
                "n_batches": args.eval_batches,
                "batch_size": args.eval_batch_size,
                "block_size": args.eval_block_size,
                "seed": args.eval_seed,
                "data": str(val_bin),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[results] {out_json}")

    print(f"\n{'='*60}\nSUMMARY  (lower is better)\n{'='*60}")
    rows = [(k, r.get("val_loss")) for k, r in results.items() if "val_loss" in r]
    rows.sort(key=lambda x: x[1] if x[1] is not None else float("inf"))
    base_loss = None
    for label, vl in rows:
        marker = "  "
        if base_loss is None and label.startswith("original/best"):
            base_loss = vl
        delta = ""
        if base_loss is not None and vl is not None and label != "original/best":
            d = vl - base_loss
            delta = f"  (vs best: {d:+.4f})"
            if d < 0:
                marker = "▼ "  # better than baseline
        print(f"  {marker}{label:<25}  val_loss = {vl:.4f}{delta}")


if __name__ == "__main__":
    main()
