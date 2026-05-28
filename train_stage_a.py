#!/usr/bin/env python3
"""
Stage A — Retrieval-Aware Continued Pretraining.

Continues from Base v2 (`final.pt`) and trains on retrieval-formatted sequences.
Mix (per sequence, weighted):
  50%  1-retrieved:    <retrieved><doc>[neighbor]</doc></retrieved>[target]
  25%  2-retrieved:    <retrieved><doc>[n1]</doc><doc>[n2]</doc></retrieved>[target]
  15%  distractor:     <retrieved><doc>[neighbor]</doc><doc>[random]</doc></retrieved>[target]
  10%  plain:          [target]

Special token IDs (from tokenizer-new-v5):
  <retrieved>=47824  </retrieved>=47825  <doc>=47826  </doc>=47827

Usage:
  python train_stage_a.py \
      --base-ckpt /workspace/telugu_lm/checkpoints/base-v2/final.pt \
      --stage-a-data /workspace/telugu_lm/stage_a_data \
      --train-bin /workspace/telugu_lm/train-data-v2/train.bin \
      --save-dir /workspace/telugu_lm/checkpoints/stage-a \
      --wandb pothana-stage-a --wandb-name stage-a-r1
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


# Special token IDs in tokenizer-new-v5
TOK_RETRIEVED_OPEN  = 47824
TOK_RETRIEVED_CLOSE = 47825
TOK_DOC_OPEN        = 47826
TOK_DOC_CLOSE       = 47827
TOK_BOS = 2
TOK_EOS = 3
TOK_PAD = 0


def import_build_model(train_gpt_path: Path):
    spec = importlib.util.spec_from_file_location("tgv2", str(train_gpt_path))
    tgv2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgv2)
    return tgv2


class RetrievalDataset:
    """Format-mixing dataset for Stage A.

    Each call to get_batch yields a (B, T) tensor of token IDs where each row
    is one of: plain, 1-retrieved, 2-retrieved, distractor.

    All sequences are packed to exactly block_size; if a formatted sequence is
    shorter, we pack additional examples after a <eos>. If longer, we truncate
    from the START of the retrieved block (keeps target intact at the end).
    """

    def __init__(self,
                 train_bin: Path,
                 chunk_offsets: np.ndarray,
                 neighbors: np.ndarray,
                 chunk_size: int,
                 block_size: int,
                 format_weights: dict,
                 seed: int = 0):
        self.data = np.memmap(str(train_bin), dtype=np.uint32, mode="r")
        self.chunk_offsets = chunk_offsets  # (N,) int64
        self.neighbors = neighbors          # (N, K) int32, -1 = no neighbor
        self.chunk_size = chunk_size
        self.block_size = block_size

        # Build format-sampling table (cumulative)
        fmt_keys = list(format_weights.keys())
        fmt_probs = np.array([format_weights[k] for k in fmt_keys], dtype=np.float64)
        fmt_probs /= fmt_probs.sum()
        self.fmt_keys = fmt_keys
        self.fmt_cum = np.cumsum(fmt_probs)
        self.rng = np.random.RandomState(seed)

    @property
    def n_chunks(self):
        return len(self.chunk_offsets)

    def _read_chunk(self, chunk_id: int) -> np.ndarray:
        off = self.chunk_offsets[chunk_id]
        return self.data[off : off + self.chunk_size].astype(np.int64)

    def _format_sequence(self, target_id: int) -> np.ndarray:
        """Return a single formatted sequence (1D int64 ndarray), up to block_size."""
        r = self.rng.random()
        idx = int(np.searchsorted(self.fmt_cum, r, side="right"))
        idx = min(idx, len(self.fmt_keys) - 1)
        fmt = self.fmt_keys[idx]

        target = self._read_chunk(target_id)

        if fmt == "plain":
            seq = np.concatenate([[TOK_BOS], target, [TOK_EOS]]).astype(np.int64)
            return seq[: self.block_size]

        # Get neighbors for retrieved formats
        nbrs = self.neighbors[target_id]
        valid_nbrs = nbrs[nbrs >= 0]
        if len(valid_nbrs) == 0:
            # Fallback: pretend it's plain
            seq = np.concatenate([[TOK_BOS], target, [TOK_EOS]]).astype(np.int64)
            return seq[: self.block_size]

        if fmt == "1-retrieved":
            n1 = self._read_chunk(int(valid_nbrs[0]))
            seq = np.concatenate([
                [TOK_BOS, TOK_RETRIEVED_OPEN, TOK_DOC_OPEN],
                n1,
                [TOK_DOC_CLOSE, TOK_RETRIEVED_CLOSE],
                target, [TOK_EOS],
            ]).astype(np.int64)
        elif fmt == "2-retrieved":
            n1 = self._read_chunk(int(valid_nbrs[0]))
            n2_id = int(valid_nbrs[1]) if len(valid_nbrs) > 1 else int(valid_nbrs[0])
            n2 = self._read_chunk(n2_id)
            seq = np.concatenate([
                [TOK_BOS, TOK_RETRIEVED_OPEN, TOK_DOC_OPEN],
                n1,
                [TOK_DOC_CLOSE, TOK_DOC_OPEN],
                n2,
                [TOK_DOC_CLOSE, TOK_RETRIEVED_CLOSE],
                target, [TOK_EOS],
            ]).astype(np.int64)
        elif fmt == "distractor":
            n1 = self._read_chunk(int(valid_nbrs[0]))
            # Random chunk (NOT a neighbor)
            rand_id = self.rng.randint(0, self.n_chunks)
            rand_chunk = self._read_chunk(rand_id)
            # Shuffle which order they appear (so model can't just pick "first chunk")
            chunks_in_order = [n1, rand_chunk] if self.rng.random() < 0.5 else [rand_chunk, n1]
            seq = np.concatenate([
                [TOK_BOS, TOK_RETRIEVED_OPEN, TOK_DOC_OPEN],
                chunks_in_order[0],
                [TOK_DOC_CLOSE, TOK_DOC_OPEN],
                chunks_in_order[1],
                [TOK_DOC_CLOSE, TOK_RETRIEVED_CLOSE],
                target, [TOK_EOS],
            ]).astype(np.int64)
        else:
            raise ValueError(f"unknown format: {fmt}")

        # Truncate: keep the TAIL (target intact); cut from the head of <retrieved>
        if len(seq) > self.block_size:
            seq = seq[-self.block_size:]
        return seq

    def get_batch(self, batch_size: int, device: str):
        """Sample batch_size sequences, pad to block_size, return (x, y)."""
        # Sample batch_size random chunk IDs (with replacement)
        target_ids = self.rng.randint(0, self.n_chunks, size=batch_size)
        seqs = [self._format_sequence(int(tid)) for tid in target_ids]

        # Pad to block_size with EOS (model learns to ignore by attending only to prior)
        B = batch_size
        T = self.block_size
        x = np.zeros((B, T), dtype=np.int64)
        y = np.zeros((B, T), dtype=np.int64)
        # We use ignore_index=-1 in CE loss for padding positions in `y`
        y.fill(-1)
        for i, seq in enumerate(seqs):
            n = len(seq)
            if n < 2:  # shouldn't happen, but safety
                continue
            n_pred = n - 1
            x[i, :n_pred] = seq[:-1]
            y[i, :n_pred] = seq[1:]
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)
        return x_t, y_t


def get_lr(step, max_steps, peak_lr, min_lr, warmup_steps, schedule="wsd",
           wsd_stable_frac=0.7, wsd_decay_frac=0.3):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if schedule == "wsd":
        stable_end = warmup_steps + int((max_steps - warmup_steps) * wsd_stable_frac)
        decay_end  = warmup_steps + int((max_steps - warmup_steps) * (wsd_stable_frac + wsd_decay_frac))
        if step < stable_end:
            return peak_lr
        if step < decay_end:
            t = (step - stable_end) / max(1, decay_end - stable_end)
            return peak_lr - (peak_lr - min_lr) * t
        return min_lr
    elif schedule == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        return peak_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path, required=True)
    ap.add_argument("--stage-a-data", type=Path, required=True)
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--train-gpt-py", type=Path, default=Path("train_gpt_v2.py"))
    # Hyperparams
    ap.add_argument("--max-steps", type=int, default=950)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lr-schedule", type=str, default="wsd", choices=["wsd", "cosine"])
    ap.add_argument("--wsd-stable-frac", type=float, default=0.6)
    ap.add_argument("--wsd-decay-frac", type=float, default=0.4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    # Mix
    ap.add_argument("--mix-1ret",       type=float, default=0.50)
    ap.add_argument("--mix-2ret",       type=float, default=0.25)
    ap.add_argument("--mix-distractor", type=float, default=0.15)
    ap.add_argument("--mix-plain",      type=float, default=0.10)
    # Misc
    ap.add_argument("--save-interval",  type=int, default=200)
    ap.add_argument("--log-interval",   type=int, default=10)
    ap.add_argument("--eval-interval",  type=int, default=100)
    ap.add_argument("--eval-batches",   type=int, default=10)
    ap.add_argument("--no-compile",     action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--wandb",          type=str, default="")
    ap.add_argument("--wandb-name",     type=str, default="")
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    # === Load Base v2 checkpoint ===
    print(f"[1/4] loading base ckpt {args.base_ckpt}", flush=True)
    tgv2 = import_build_model(args.train_gpt_py)
    ckpt = torch.load(str(args.base_ckpt), map_location="cpu", weights_only=False, mmap=True)
    cfg_kwargs = {k: v for k, v in ckpt["config"].items() if k in tgv2.GPTConfig.__dataclass_fields__}
    cfg = tgv2.GPTConfig(**cfg_kwargs)
    model = tgv2.build_model(cfg, device=device).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    non_freqs = [k for k in missing if "freqs_cis" not in k]
    print(f"       loaded — missing(non-freqs)={len(non_freqs)} unexpected={len(unexpected)}")
    print(f"       params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M  block_size={cfg.block_size}")
    if args.grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("       grad checkpointing: ON")
    del ckpt

    # === Load Stage A data ===
    print(f"[2/4] loading Stage A data {args.stage_a_data}", flush=True)
    chunk_offsets = np.load(args.stage_a_data / "chunk_offsets.npy")
    neighbors = np.load(args.stage_a_data / "neighbors.npy")
    meta = json.load(open(args.stage_a_data / "meta.json"))
    chunk_size = meta["chunk_size"]
    print(f"       {len(chunk_offsets):,} chunks, chunk_size={chunk_size}")
    print(f"       mean valid neighbors/chunk: {meta['stats']['valid_neighbors_per_chunk_mean']:.2f}")

    mix_weights = {
        "1-retrieved": args.mix_1ret,
        "2-retrieved": args.mix_2ret,
        "distractor": args.mix_distractor,
        "plain": args.mix_plain,
    }
    print(f"       format mix: {mix_weights}")
    dataset = RetrievalDataset(
        train_bin=args.train_bin,
        chunk_offsets=chunk_offsets,
        neighbors=neighbors,
        chunk_size=chunk_size,
        block_size=cfg.block_size,
        format_weights=mix_weights,
        seed=42,
    )

    # === Optimizer ===
    print(f"[3/4] optimizer + schedule", flush=True)
    # No weight decay on norms or 1D params
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() < 2 or "norm" in n.lower() or "ln_" in n.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr, betas=(args.beta1, args.beta2), fused=True)

    # === wandb ===
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb, name=args.wandb_name or None,
                               config={**vars(args), "n_params": sum(p.numel() for p in model.parameters())})
        print(f"       wandb: {wandb_run.url}", flush=True)

    print(f"[4/4] training {args.max_steps} steps", flush=True)
    print(f"       effective batch: {args.batch_size} x {args.grad_accum} = {args.batch_size*args.grad_accum} seqs")
    print(f"       tokens/step: {args.batch_size*args.grad_accum*cfg.block_size:,}")
    print(f"       total tokens: {args.batch_size*args.grad_accum*cfg.block_size*args.max_steps/1e9:.2f}B")

    # === Compile ===
    if not args.no_compile:
        print("       compiling model...", flush=True)
        model = torch.compile(model)

    # === Train loop ===
    t_start = time.time()
    best_val = float("inf")
    model.train()
    for step in range(args.max_steps):
        lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps,
                    args.lr_schedule, args.wsd_stable_frac, args.wsd_decay_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulate gradients
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = dataset.get_batch(args.batch_size, device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tokens_per_step = args.batch_size * args.grad_accum * cfg.block_size
            tps = tokens_per_step * (step + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step:4d}/{args.max_steps} | loss {total_loss:.4f} | lr {lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.1f} min", flush=True)
            if wandb_run:
                wandb_run.log({"train/loss": total_loss, "train/lr": lr, "train/tok_s": tps, "step": step})

        if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps:
            # Quick eval: a few batches of mixed-format data
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(args.eval_batches):
                    x, y = dataset.get_batch(args.batch_size, device)
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, eloss = model(x, targets=y)
                    losses.append(eloss.item())
                val_loss = float(np.mean(losses))
            print(f"  eval at step {step+1}: val_loss={val_loss:.4f}", flush=True)
            if wandb_run:
                wandb_run.log({"val/loss": val_loss, "step": step + 1})
            if val_loss < best_val:
                best_val = val_loss
                _save(model, optimizer, cfg, step + 1, val_loss, best_val, args.save_dir / "best.pt")
            model.train()

        if (step + 1) % args.save_interval == 0:
            _save(model, optimizer, cfg, step + 1, total_loss, best_val, args.save_dir / f"step_{step+1:05d}.pt")

    _save(model, optimizer, cfg, args.max_steps, total_loss, best_val, args.save_dir / "final.pt")
    print(f"\nDone. {(time.time()-t_start)/3600:.2f}h. best_val={best_val:.4f}")


def _save(model, optimizer, cfg, step, loss, best_val, path):
    # Unwrap torch.compile if present
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = {
        "model": m.state_dict(),
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "architecture": "llama",
        "step": step,
        "val_loss": loss,
        "best_val_loss": best_val,
    }
    torch.save(state, str(path))
    print(f"    saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
