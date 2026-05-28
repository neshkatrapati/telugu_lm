#!/usr/bin/env python3
"""
Build train-codemix.bin for Stage A+ from codemix JSONL files + train-data-v2.

Mix (per sequence):
  30% codemix_te_en   (Telugu-script + English-script mix)
  30% codemix_roman   (all Roman codemix)
  30% telugu_roman    (pure Roman Telugu)
  10% telugu original (anti-forgetting buffer from train-data-v2/train.bin)

Tokenizes with the new v6 tokenizer (which has direct tokens for top-5K
Roman Telugu words), so Roman content becomes much more efficient.

Usage:
    python build_codemix_train.py \\
        --inputs codemix-10k.jsonl codemix-5k-b2.jsonl codemix-5k-b2-partial-24w.jsonl \\
        --tokenizer tokenizer-new-v6 \\
        --train-bin train-data-v2/train.bin \\
        --out-dir codemix-train-data \\
        --val-frac 0.01 \\
        --workers 8
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from collections import Counter

import numpy as np


# Format mix
DEFAULT_MIX = {
    "codemix_te_en": 0.30,
    "codemix_roman": 0.30,
    "telugu_roman":  0.30,
    "telugu_orig":   0.10,
}


# Global state per worker
_GLOBAL = {}


def _worker_init(tok_dir):
    sys.path.insert(0, "/workspace/telugu_lm")
    from train_tokenizer import MorfessorTokenizer
    _GLOBAL["tok"] = MorfessorTokenizer(tok_dir)


def encode_text(text):
    """Encode text via MorfessorTokenizer (v6 has direct tokens for top-5K Roman)."""
    tok = _GLOBAL["tok"]
    return tok.encode(text, add_bos=False, add_eos=True)


def _process(args):
    """args = (label, text). Returns list[int] of token IDs."""
    label, text = args
    if not text:
        return []
    try:
        return encode_text(text)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="codemix JSONL files")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--train-bin", type=Path, required=True,
                    help="Source train.bin to sample original Telugu chunks from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-size-orig", type=int, default=512,
                    help="Token-size of plain Telugu chunks sampled from train.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: collect codemix records -----
    print(f"[1/4] collecting codemix records from {len(args.inputs)} JSONL files", flush=True)
    records = []
    for path in args.inputs:
        n_ok = 0
        for line in open(path):
            r = json.loads(line)
            if "_error" in r or "_skip" in r: continue
            records.append({
                "codemix_te_en": r.get("codemix_te_en") or "",
                "codemix_roman": r.get("codemix_roman") or "",
                "telugu_roman":  r.get("telugu_roman")  or "",
            })
            n_ok += 1
        print(f"   {path.name}: {n_ok} ok")
    print(f"   total: {len(records):,} codemix records")

    # ----- Step 2: build the work list — three formats per record + sampled originals -----
    # For each record, emit one sequence per non-empty format
    work = []
    for r in records:
        for label in ("codemix_te_en", "codemix_roman", "telugu_roman"):
            text = r.get(label, "")
            if text:
                work.append((label, text))

    # Sample original Telugu chunks from train.bin to hit the 10% mix
    # If codemix gives 90% of total, we need 10% from train.bin
    rng = np.random.RandomState(args.seed)
    n_codemix = len(work)
    # We want telugu_orig to be 10% / 90% × n_codemix in count (approx)
    n_orig = int(n_codemix * (DEFAULT_MIX["telugu_orig"] / (1 - DEFAULT_MIX["telugu_orig"])))
    print(f"   codemix sequences: {n_codemix:,}")
    print(f"   sampling {n_orig:,} plain Telugu chunks from {args.train_bin}", flush=True)

    train_mem = np.memmap(str(args.train_bin), dtype=np.uint32, mode="r")
    max_start = len(train_mem) - args.chunk_size_orig - 1
    orig_offsets = rng.randint(0, max_start, size=n_orig).astype(np.int64)

    # ----- Step 3: encode in parallel (codemix) + read direct (original) -----
    print(f"[2/4] encoding codemix sequences with {args.workers} workers", flush=True)
    rng.shuffle(work)  # shuffle codemix order
    t0 = time.time()

    out_train = args.out_dir / "train.bin"
    out_val   = args.out_dir / "val.bin"
    # Buffer to track all tokens before splitting train/val
    tmp_bin = args.out_dir / "all.tmp.bin"

    total_codemix_tokens = 0
    label_counts = Counter()
    label_tokens = Counter()
    BATCH = 1024
    n_done = 0
    last_log = t0

    with open(tmp_bin, "wb") as fout:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(str(args.tokenizer),)) as pool:
            # Process codemix in batches
            for i in range(0, len(work), BATCH):
                batch = work[i : i + BATCH]
                results = pool.map(_process, batch)
                for (label, _), ids in zip(batch, results):
                    if not ids: continue
                    arr = np.array(ids, dtype=np.uint32)
                    fout.write(arr.tobytes())
                    total_codemix_tokens += len(ids)
                    label_counts[label] += 1
                    label_tokens[label] += len(ids)
                    n_done += 1
                if time.time() - last_log >= 15:
                    elapsed = time.time() - t0
                    rate = n_done / elapsed
                    pct = 100 * n_done / len(work)
                    eta_min = (len(work) - n_done) / max(rate, 1e-9) / 60
                    print(f"   [{pct:5.1f}%] {n_done:>7,}/{len(work):,}  {rate:>5.0f}/s  ETA {eta_min:.1f}m", flush=True)
                    last_log = time.time()
    print(f"   encoded {n_done:,} codemix sequences in {(time.time()-t0)/60:.1f}m")
    print(f"   total codemix tokens: {total_codemix_tokens:,}")

    # ----- Step 4: append plain Telugu original chunks -----
    print(f"[3/4] appending {len(orig_offsets):,} plain Telugu chunks from train.bin", flush=True)
    EOS = 3
    n_orig_tokens = 0
    with open(tmp_bin, "ab") as fout:
        for off in orig_offsets:
            ids = train_mem[off : off + args.chunk_size_orig].tolist()
            ids.append(EOS)
            arr = np.array(ids, dtype=np.uint32)
            fout.write(arr.tobytes())
            n_orig_tokens += len(ids)
    label_counts["telugu_orig"] = len(orig_offsets)
    label_tokens["telugu_orig"] = n_orig_tokens
    print(f"   plain Telugu tokens: {n_orig_tokens:,}")

    total_tokens = total_codemix_tokens + n_orig_tokens
    print(f"   GRAND TOTAL: {total_tokens:,} tokens ({total_tokens/1e6:.1f}M, {total_tokens/1e9:.2f}B)")

    # ----- Step 5: split train/val -----
    print(f"[4/4] splitting train/val (val_frac={args.val_frac})", flush=True)
    n_total = os.path.getsize(tmp_bin) // 4
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    mmap = np.memmap(str(tmp_bin), dtype=np.uint32, mode="r")
    mmap[:n_train].tofile(str(out_train))
    mmap[n_train:].tofile(str(out_val))
    del mmap
    tmp_bin.unlink()

    # ----- Save meta -----
    meta = {
        "tokenizer": str(args.tokenizer),
        "source_train_bin": str(args.train_bin),
        "vocab_size": 52831,
        "dtype": "uint32",
        "format_mix": {label: {"count": label_counts[label], "tokens": label_tokens[label]}
                       for label in label_counts},
        "n_codemix_records": len(records),
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "total_tokens": int(n_total),
        "format_mix_target": DEFAULT_MIX,
        "notes": "Codemix data from Gemini 2.0 Flash + plain Telugu from train-data-v2. Tokenized with v6 (47K + 5K Roman tokens).",
    }
    with open(args.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output: {args.out_dir}")
    print(f"  train.bin: {os.path.getsize(out_train)/1e9:.2f} GB ({n_train:,} tokens)")
    print(f"  val.bin:   {os.path.getsize(out_val)/1e6:.1f} MB ({n_val:,} tokens)")
    print(f"\nFormat mix achieved (by tokens):")
    for label, n_tok in label_tokens.most_common():
        pct = 100 * n_tok / total_tokens
        print(f"  {label:<20s} {label_counts[label]:>7,} seqs  {n_tok/1e6:>6.1f}M tokens  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
