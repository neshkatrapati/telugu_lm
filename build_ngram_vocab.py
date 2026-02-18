#!/usr/bin/env python3
"""
Build N-gram Vocabulary for Engram
===================================
Scans tokenized training shards (uint32 .bin files from train_gpt.py prepare)
and builds an exact N-gram vocabulary with direct index assignment.

Output: ngram_vocab.json containing:
  - bigrams:  {"tok_a,tok_b": index, ...}     — known bigrams with assigned indices
  - trigrams: {"tok_a,tok_b,tok_c": index, ...} — known trigrams with assigned indices
  - bigram_count, trigram_count
  - min_count, stats

Known N-grams get unique indices (1-indexed). Unknown N-grams map to index 0
(UNK). The unigram table (in the Engram module) always provides a valid signal,
so missing bi/trigrams are not a problem.

Usage:
    python build_ngram_vocab.py --data ./train_data --output ./tokenizer/ngram_vocab.json --min-count 50
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── N-gram counting ─────────────────────────────────────────────────────────

def count_ngrams(data_path: Path, min_count: int):
    """
    Count bigram and trigram frequencies from a uint32 memmap file.

    Strategy: stream through data in chunks, build packed uint64 keys for each
    N-gram, and count them using numpy. The key insight is that we DON'T store
    per-chunk dicts and merge them — instead we accumulate all keys across chunks,
    then do a single global np.unique at the end. This avoids the RAM explosion
    from Python dict merging.

    For very large files where even storing all keys would be too much RAM,
    we use a two-pass approach: first pass counts with numpy sort+diff on
    chunk-sized arrays, writing intermediate (key, count) pairs to a temp file,
    then second pass merges.

    But the simplest approach that works for our scale (~500M tokens):
    - Bigram keys: 500M uint64 values = 4 GB RAM
    - Trigram keys: 500M uint64 values = 4 GB RAM
    That's too much. So we chunk and use a single Python Counter, but feed it
    efficiently using numpy's unique within each chunk (far fewer unique keys
    than raw keys).
    """
    logger.info("Loading %s ...", data_path)
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    n_tokens = len(data)
    logger.info("  %d tokens (%.2f GB)", n_tokens, n_tokens * 4 / 1e9)

    # Pack N-gram token IDs into single int64 keys
    # With vocab ~86K, MAX_TOKEN = 100000 is safe (86K^3 = 6.4e14 < 2^63)
    MAX_TOKEN = np.int64(100_000)

    # Use smaller chunks to keep peak RAM low
    # Each chunk: 10M tokens → ~80 MB for the int64 copy + ~80 MB for keys
    CHUNK = 10_000_000

    # ── Pass 1: Count bigrams ──
    logger.info("Pass 1/2: Counting bigrams ...")
    bigram_counts = {}
    with tqdm(total=n_tokens, desc="Bigrams", unit="tok", unit_scale=True) as pbar:
        for start in range(0, n_tokens - 1, CHUNK):
            end = min(start + CHUNK + 1, n_tokens)  # +1 overlap for pairs
            chunk = data[start:end].astype(np.int64)
            own_len = min(CHUNK, n_tokens - 1 - start)

            keys = chunk[:own_len] * MAX_TOKEN + chunk[1:own_len + 1]
            unique_keys, counts = np.unique(keys, return_counts=True)

            # Merge into global dict — unique_keys per chunk is much smaller than raw
            for k, c in zip(unique_keys, counts):
                k = int(k)
                bigram_counts[k] = bigram_counts.get(k, 0) + int(c)

            del chunk, keys, unique_keys, counts
            pbar.update(own_len)

    logger.info("  Unique bigrams: %d", len(bigram_counts))

    # ── Pass 2: Count trigrams ──
    logger.info("Pass 2/2: Counting trigrams ...")
    trigram_counts = {}
    with tqdm(total=n_tokens, desc="Trigrams", unit="tok", unit_scale=True) as pbar:
        for start in range(0, n_tokens - 2, CHUNK):
            end = min(start + CHUNK + 2, n_tokens)  # +2 overlap for triples
            chunk = data[start:end].astype(np.int64)
            own_len = min(CHUNK, n_tokens - 2 - start)

            keys = (chunk[:own_len] * (MAX_TOKEN * MAX_TOKEN)
                    + chunk[1:own_len + 1] * MAX_TOKEN
                    + chunk[2:own_len + 2])
            unique_keys, counts = np.unique(keys, return_counts=True)

            for k, c in zip(unique_keys, counts):
                k = int(k)
                trigram_counts[k] = trigram_counts.get(k, 0) + int(c)

            del chunk, keys, unique_keys, counts
            pbar.update(own_len)

    logger.info("  Unique trigrams: %d", len(trigram_counts))

    # ── Apply cutoff ──
    logger.info("Applying min_count=%d cutoff ...", min_count)
    bigrams = []
    bi_total = 0
    bi_kept_freq = 0
    for key, count in tqdm(bigram_counts.items(), desc="Filtering bigrams", unit="ngram"):
        bi_total += count
        if count >= min_count:
            a = int(key // MAX_TOKEN)
            b = int(key % MAX_TOKEN)
            bigrams.append((a, b))
            bi_kept_freq += count

    del bigram_counts

    trigrams = []
    tri_total = 0
    tri_kept_freq = 0
    for key, count in tqdm(trigram_counts.items(), desc="Filtering trigrams", unit="ngram"):
        tri_total += count
        if count >= min_count:
            a = int(key // (MAX_TOKEN * MAX_TOKEN))
            rem = int(key % (MAX_TOKEN * MAX_TOKEN))
            b = int(rem // MAX_TOKEN)
            c = int(rem % MAX_TOKEN)
            trigrams.append((a, b, c))
            tri_kept_freq += count

    del trigram_counts

    bigrams.sort()
    trigrams.sort()

    logger.info("After min_count=%d cutoff:", min_count)
    logger.info("  Bigrams:  %d", len(bigrams))
    logger.info("  Trigrams: %d", len(trigrams))
    logger.info("  Bigram coverage:  %.2f%% of all bigram occurrences",
                100 * bi_kept_freq / bi_total if bi_total else 0)
    logger.info("  Trigram coverage: %.2f%% of all trigram occurrences",
                100 * tri_kept_freq / tri_total if tri_total else 0)

    stats = {
        "n_tokens": int(n_tokens),
        "unique_bigrams_total": int(bi_total),  # this is occurrence count
        "unique_trigrams_total": int(tri_total),
        "bigram_occurrences_total": int(bi_total),
        "trigram_occurrences_total": int(tri_total),
        "bigrams_after_cutoff": len(bigrams),
        "trigrams_after_cutoff": len(trigrams),
        "bigram_coverage_pct": round(100 * bi_kept_freq / bi_total, 2) if bi_total else 0,
        "trigram_coverage_pct": round(100 * tri_kept_freq / tri_total, 2) if tri_total else 0,
    }

    return bigrams, trigrams, stats


# ── Index assignment ─────────────────────────────────────────────────────────

def build_ngram_index(bigrams, trigrams):
    """
    Assign sequential indices to known N-grams.

    Layout for each table (bigram / trigram):
        [0]        → UNK embedding (for unknown N-grams)
        [1 ... N]  → known N-grams (exact, zero collisions)
    """
    bigram_to_idx = {}
    for i, ng in enumerate(tqdm(bigrams, desc="Indexing bigrams", unit="ngram")):
        bigram_to_idx[ng] = i + 1  # 1-indexed, 0 = UNK

    trigram_to_idx = {}
    for i, ng in enumerate(tqdm(trigrams, desc="Indexing trigrams", unit="ngram")):
        trigram_to_idx[ng] = i + 1

    bigram_table_size = 1 + len(bigrams)
    trigram_table_size = 1 + len(trigrams)

    logger.info("Index assignment:")
    logger.info("  Bigram:  %d known + 1 UNK = %d total rows",
                len(bigrams), bigram_table_size)
    logger.info("  Trigram: %d known + 1 UNK = %d total rows",
                len(trigrams), trigram_table_size)

    return bigram_to_idx, trigram_to_idx, bigram_table_size, trigram_table_size


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build N-gram vocabulary with exact indices for Engram"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to directory containing train.bin (from prepare step)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for ngram_vocab.json")
    parser.add_argument("--min-count", type=int, default=50,
                        help="Minimum frequency to include an N-gram (default: 50)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_bin = data_dir / "train.bin"

    if not train_bin.exists():
        logger.error("train.bin not found in %s", data_dir)
        logger.error("Run `python train_gpt.py prepare` first.")
        sys.exit(1)

    # ── Step 1: Count N-grams ──
    bigrams, trigrams, stats = count_ngrams(train_bin, args.min_count)

    if len(bigrams) == 0 and len(trigrams) == 0:
        logger.error("No N-grams survived the cutoff! Try lowering --min-count")
        sys.exit(1)

    # ── Step 2: Assign indices ──
    bigram_to_idx, trigram_to_idx, bi_table_size, tri_table_size = build_ngram_index(
        bigrams, trigrams
    )

    # ── Step 3: Save ──
    logger.info("Serializing to JSON ...")
    bigram_dict = {f"{a},{b}": idx for (a, b), idx in
                   tqdm(bigram_to_idx.items(), desc="Serializing bigrams", unit="ngram")}
    trigram_dict = {f"{a},{b},{c}": idx for (a, b, c), idx in
                    tqdm(trigram_to_idx.items(), desc="Serializing trigrams", unit="ngram")}

    output = {
        "version": 2,
        "min_count": args.min_count,

        "bigram_count": len(bigrams),
        "bigram_table_size": bi_table_size,
        "trigram_count": len(trigrams),
        "trigram_table_size": tri_table_size,

        "bigrams": bigram_dict,
        "trigrams": trigram_dict,

        "stats": stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing %s ...", output_path)
    with open(output_path, "w") as f:
        json.dump(output, f)

    file_size = os.path.getsize(output_path)
    logger.info("Saved N-gram vocab to %s (%.1f MB)", output_path, file_size / 1e6)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("N-gram Vocabulary Summary")
    logger.info("=" * 60)
    logger.info("  Source:          %s", train_bin)
    logger.info("  Tokens:          %d", stats["n_tokens"])
    logger.info("  Min count:       %d", args.min_count)
    logger.info("  Bigrams:         %d (coverage: %.1f%%)",
                len(bigrams), stats["bigram_coverage_pct"])
    logger.info("  Trigrams:        %d (coverage: %.1f%%)",
                len(trigrams), stats["trigram_coverage_pct"])
    logger.info("  Bigram table:    %d rows (%d known + 1 UNK)",
                bi_table_size, len(bigrams))
    logger.info("  Trigram table:   %d rows (%d known + 1 UNK)",
                tri_table_size, len(trigrams))
    logger.info("")

    # Param estimates
    vocab_size = 86075  # approximate
    d = 64  # per-order dimension
    uni_params = vocab_size * d
    bi_params = bi_table_size * d
    tri_params = tri_table_size * d
    total = uni_params + bi_params + tri_params
    logger.info("  Estimated Engram params (at dim=%d per order):", d)
    logger.info("    Unigram table: %d x %d = %.1fM params", vocab_size, d, uni_params / 1e6)
    logger.info("    Bigram table:  %d x %d = %.1fM params", bi_table_size, d, bi_params / 1e6)
    logger.info("    Trigram table: %d x %d = %.1fM params", tri_table_size, d, tri_params / 1e6)
    logger.info("    Shared total:  %.1fM params", total / 1e6)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
