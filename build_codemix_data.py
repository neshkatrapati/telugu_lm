#!/usr/bin/env python3
"""
Stage A+ data builder — Roman Telugu (Tenglish) continued-pretrain corpus.

Reads chunks from train-data-v2/train.bin, decodes to text, transliterates
Telugu portions to phone-style Roman Telugu (Velthuis + casualize), re-tokenizes,
and writes a mixed-format binary for continued pretraining.

Mix (target ~500M tokens):
  60% Roman Telugu monolingual         — exposes the model to Roman script
  25% Parallel pairs (TE then ROM)     — explicit script-equivalence signal
  15% Plain Telugu                     — anti-forgetting buffer

Mixed Telugu+English chunks pass English through unchanged (aksharamukha
preserves Latin characters).

Usage:
    python build_codemix_data.py \\
        --train-bin /workspace/telugu_lm/train-data-v2/train.bin \\
        --tokenizer-src /workspace/telugu_lm/tokenizer-new-v5 \\
        --tokenizer-dir /workspace/telugu_lm/tokenizer \\
        --out-dir /workspace/telugu_lm/codemix-data \\
        --n-chunks 600000 \\
        --chunk-size 512 \\
        --workers 8
"""
import argparse
import json
import os
import re
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np


# ---------- Token bookkeeping ----------

TOK_BOS = 2
TOK_EOS = 3
TOK_PAD = 0
TOK_UNK = 1

SPECIAL_TOKEN_STRS = {"<pad>", "<unk>", "<bos>", "<eos>",
                       "<search>", "</search>", "<retrieved>", "</retrieved>",
                       "<doc>", "</doc>", "<cite>", "<think>", "</think>"}


def load_inverse_vocab(tokenizer_json: Path) -> dict:
    """id → token string."""
    data = json.load(open(tokenizer_json))
    return {v: k for k, v in data["token_to_id"].items()}


def decode_chunk(ids, id_to_token):
    """Decode token IDs to clean text (drop specials, strip @@)."""
    parts = []
    for tid in ids:
        tok = id_to_token.get(int(tid))
        if tok is None or tok in SPECIAL_TOKEN_STRS:
            continue
        parts.append(tok)
    text = " ".join(parts)
    # @@ continuation prefix — join to previous
    text = text.replace(" @@", "").replace("@@", "")
    return text.strip()


# ---------- Roman Telugu generation ----------

_RETROFLEX_PAT = re.compile(r"\.([dtnsmlNTMRr])")

def casualize_velthuis(s: str) -> str:
    """Make Velthuis output look like what people actually type on phones."""
    # Drop retroflex / nasal dots: .d → d, .t → t, .m → m, etc.
    s = _RETROFLEX_PAT.sub(r"\1", s)
    # Lowercase
    s = s.lower()
    # Compress whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- Worker process ----------

_GLOBAL = {}


def _worker_init(train_bin_path: str, tok_src_path: str, tok_dir_path: str):
    """Each worker loads its own copies of memmap + tokenizer."""
    global _GLOBAL
    sys.path.insert(0, "/workspace/telugu_lm")
    from train_tokenizer import MorfessorTokenizer
    from aksharamukha import transliterate

    _GLOBAL["data"] = np.memmap(train_bin_path, dtype=np.uint32, mode="r")
    _GLOBAL["id_to_token"] = load_inverse_vocab(Path(tok_src_path) / "tokenizer.json")
    _GLOBAL["tok"] = MorfessorTokenizer(tok_dir_path)
    _GLOBAL["transliterate"] = transliterate


def _process_chunk(args):
    """
    args = (chunk_offset, chunk_size, format_label)
    format_label ∈ {"roman", "parallel", "telugu"}
    Returns flat list[int] of token IDs to append to output.
    """
    offset, chunk_size, fmt = args
    data = _GLOBAL["data"]
    id_to_token = _GLOBAL["id_to_token"]
    tok = _GLOBAL["tok"]
    transliterate = _GLOBAL["transliterate"]

    raw_ids = data[offset : offset + chunk_size].tolist()
    text = decode_chunk(raw_ids, id_to_token)
    if not text:
        return []

    if fmt == "telugu":
        # Re-tokenize the cleaned text (no transliteration)
        ids = tok.encode(text, add_bos=False, add_eos=True)
        return ids

    # Transliterate (handles Telugu chars; passes English/Latin through)
    try:
        rom = transliterate.process("Telugu", "Velthuis", text)
        rom = casualize_velthuis(rom)
    except Exception:
        return []

    if not rom or rom == text:
        return []

    if fmt == "roman":
        ids = tok.encode(rom, add_bos=False, add_eos=True)
        return ids
    elif fmt == "parallel":
        # TE then ROM, separated by a regular space (no special separator —
        # the model learns alignment from co-occurrence)
        combined = f"{text} {rom}"
        ids = tok.encode(combined, add_bos=False, add_eos=True)
        return ids

    return []


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--tokenizer-src", type=Path, required=True,
                    help="Source tokenizer dir with tokenizer.json (token_to_id format)")
    ap.add_argument("--tokenizer-dir", type=Path, required=True,
                    help="Tokenizer dir with vocab.txt + BPE merges (for MorfessorTokenizer)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-chunks", type=int, default=600_000)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--mix-roman",    type=float, default=0.60)
    ap.add_argument("--mix-parallel", type=float, default=0.25)
    ap.add_argument("--mix-telugu",   type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Sample chunk offsets ----------
    print(f"[1/4] sampling chunk offsets", flush=True)
    n_tokens = args.train_bin.stat().st_size // 4
    rng = np.random.RandomState(args.seed)
    max_start = n_tokens - args.chunk_size - 1
    stride = max_start // args.n_chunks
    base = np.arange(args.n_chunks, dtype=np.int64) * stride
    jitter = rng.randint(0, max(1, stride), size=args.n_chunks).astype(np.int64)
    chunk_offsets = (base + jitter).astype(np.int64)
    print(f"      sampled {len(chunk_offsets):,} chunks (stride={stride:,})", flush=True)

    # ---------- Assign formats ----------
    print(f"[2/4] assigning formats: roman={args.mix_roman:.0%} parallel={args.mix_parallel:.0%} telugu={args.mix_telugu:.0%}", flush=True)
    # Cumulative thresholds
    probs = np.array([args.mix_roman, args.mix_parallel, args.mix_telugu])
    probs = probs / probs.sum()
    labels = ["roman", "parallel", "telugu"]
    r = rng.random(args.n_chunks)
    formats = []
    cum = np.cumsum(probs)
    for v in r:
        idx = int(np.searchsorted(cum, v, side="right"))
        idx = min(idx, len(labels) - 1)
        formats.append(labels[idx])
    from collections import Counter
    fmt_counts = Counter(formats)
    for k, v in fmt_counts.items():
        print(f"      {k}: {v:,}", flush=True)

    # ---------- Process chunks in parallel ----------
    print(f"[3/4] processing with {args.workers} workers", flush=True)
    work = [(int(off), args.chunk_size, fmt) for off, fmt in zip(chunk_offsets, formats)]

    out_bin = args.out_dir / "train_codemix.bin"
    total_tokens = 0
    n_done = 0
    n_empty = 0
    t0 = time.time()
    last_log = t0
    BATCH = 2048

    with open(out_bin, "wb") as fout:
        with mp.Pool(args.workers,
                     initializer=_worker_init,
                     initargs=(str(args.train_bin), str(args.tokenizer_src), str(args.tokenizer_dir))) as pool:
            for batch_start in range(0, len(work), BATCH):
                batch = work[batch_start : batch_start + BATCH]
                results = pool.map(_process_chunk, batch)
                for ids in results:
                    n_done += 1
                    if not ids:
                        n_empty += 1
                        continue
                    arr = np.array(ids, dtype=np.uint32)
                    fout.write(arr.tobytes())
                    total_tokens += len(ids)
                if time.time() - last_log >= 30:
                    elapsed = time.time() - t0
                    pct = 100 * n_done / args.n_chunks
                    rate = n_done / elapsed
                    eta_min = (args.n_chunks - n_done) / max(rate, 1) / 60
                    print(f"      [{pct:5.1f}%] {n_done:>7,}/{args.n_chunks:,}  "
                          f"{rate:>5.0f} chunks/s  tokens={total_tokens/1e6:>7.1f}M  ETA {eta_min:5.1f}m  empty={n_empty}",
                          flush=True)
                    last_log = time.time()
    elapsed = time.time() - t0
    print(f"      done in {elapsed/60:.1f} min", flush=True)
    print(f"      tokens written: {total_tokens:,} ({total_tokens/1e6:.1f}M, {total_tokens/1e9:.2f}B)", flush=True)
    print(f"      empty chunks (skipped): {n_empty:,}", flush=True)

    # ---------- Save val split (last 1%) and meta ----------
    print(f"[4/4] saving meta + val split", flush=True)
    out_size = out_bin.stat().st_size
    n_total_tokens = out_size // 4
    val_tokens = int(n_total_tokens * 0.01)
    train_tokens = n_total_tokens - val_tokens
    print(f"      train: {train_tokens:,}  val: {val_tokens:,}", flush=True)

    # Split via memmap
    out_train = args.out_dir / "train.bin"
    out_val = args.out_dir / "val.bin"
    mmap = np.memmap(str(out_bin), dtype=np.uint32, mode="r")
    mmap[:train_tokens].tofile(str(out_train))
    mmap[train_tokens:].tofile(str(out_val))
    del mmap
    out_bin.unlink()  # delete the unsplit file

    meta = {
        "source_train_bin": str(args.train_bin),
        "tokenizer_src": str(args.tokenizer_src),
        "tokenizer_dir": str(args.tokenizer_dir),
        "vocab_size": 47831,
        "dtype": "uint32",
        "n_chunks_requested": args.n_chunks,
        "n_chunks_empty": n_empty,
        "chunk_size": args.chunk_size,
        "format_mix": {
            "roman": args.mix_roman,
            "parallel": args.mix_parallel,
            "telugu": args.mix_telugu,
        },
        "format_counts": dict(fmt_counts),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "total_tokens": int(n_total_tokens),
        "build_time_min": elapsed / 60,
        "transliteration": "aksharamukha Velthuis + casualize (lowercase + drop retroflex dots)",
        "notes": (
            "Roman Telugu generated via aksharamukha Velthuis scheme, post-processed with "
            "casualize() to drop retroflex marks (.d→d) and lowercase. Mixed Telugu+English "
            "input has English/Latin passed through unchanged."
        ),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"      output: {args.out_dir}/")
    for p in sorted(args.out_dir.iterdir()):
        sz = p.stat().st_size
        print(f"        {p.name}  ({sz/1e9:.2f} GB)" if sz > 1e9 else f"        {p.name}  ({sz/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
