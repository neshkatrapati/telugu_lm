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
from collections import Counter

import numpy as np

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
    Uses vectorized numpy for speed.

    Returns:
        bigrams:  sorted list of (token_a, token_b) tuples that meet min_count
        trigrams: sorted list of (token_a, token_b, token_c) tuples that meet min_count
        stats:    dict with counting statistics
    """
    logger.info("Loading %s ...", data_path)
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    n_tokens = len(data)
    logger.info("  %d tokens (%.2f GB)", n_tokens, n_tokens * 4 / 1e9)

    logger.info("Counting N-grams (vectorized) ...")
    t0 = time.time()

    # Pack N-gram token IDs into single uint64 keys for fast counting
    # With vocab ~86K, MAX_TOKEN = 100000 is safe (86K^3 = 6.4e14 < 2^64 = 1.8e19)
    MAX_TOKEN = 100_000

    bigram_counts = Counter()
    trigram_counts = Counter()

    # Process in chunks (numpy vectorized within each chunk)
    CHUNK = 50_000_000  # 50M tokens per chunk

    for start in range(0, n_tokens, CHUNK):
        end = min(start + CHUNK + 2, n_tokens)
        chunk = np.array(data[start:end], dtype=np.int64)

        bi_len = min(len(chunk) - 1, CHUNK if start + CHUNK < n_tokens else len(chunk) - 1)

        # Vectorized bigram keys
        bi_keys = chunk[:bi_len] * MAX_TOKEN + chunk[1:bi_len + 1]
        unique_bi, counts_bi = np.unique(bi_keys, return_counts=True)
        for k, c in zip(unique_bi.tolist(), counts_bi.tolist()):
            bigram_counts[k] += c

        # Vectorized trigram keys
        tri_len = min(len(chunk) - 2, CHUNK if start + CHUNK < n_tokens else len(chunk) - 2)
        tri_keys = (chunk[:tri_len] * (MAX_TOKEN * MAX_TOKEN)
                    + chunk[1:tri_len + 1] * MAX_TOKEN
                    + chunk[2:tri_len + 2])
        unique_tri, counts_tri = np.unique(tri_keys, return_counts=True)
        for k, c in zip(unique_tri.tolist(), counts_tri.tolist()):
            trigram_counts[k] += c

        elapsed = time.time() - t0
        pct = 100 * min(start + CHUNK, n_tokens) / n_tokens
        logger.info("  %.1f%% done (%d/%d tokens, %.0fs)",
                    pct, min(start + CHUNK, n_tokens), n_tokens, elapsed)

    elapsed = time.time() - t0
    logger.info("Counting done in %.1fs", elapsed)
    logger.info("  Unique bigrams:  %d", len(bigram_counts))
    logger.info("  Unique trigrams: %d", len(trigram_counts))

    # Decode keys back to tuples and apply cutoff
    bigrams = []
    bi_total = 0
    bi_kept_freq = 0
    for key, count in bigram_counts.items():
        bi_total += count
        if count >= min_count:
            a = int(key // MAX_TOKEN)
            b = int(key % MAX_TOKEN)
            bigrams.append((a, b))
            bi_kept_freq += count

    trigrams = []
    tri_total = 0
    tri_kept_freq = 0
    for key, count in trigram_counts.items():
        tri_total += count
        if count >= min_count:
            a = int(key // (MAX_TOKEN * MAX_TOKEN))
            rem = int(key % (MAX_TOKEN * MAX_TOKEN))
            b = int(rem // MAX_TOKEN)
            c = int(rem % MAX_TOKEN)
            trigrams.append((a, b, c))
            tri_kept_freq += count

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
        "unique_bigrams_total": len(bigram_counts),
        "unique_trigrams_total": len(trigram_counts),
        "bigram_occurrences_total": int(bi_total),
        "trigram_occurrences_total": int(tri_total),
        "bigrams_after_cutoff": len(bigrams),
        "trigrams_after_cutoff": len(trigrams),
        "bigram_coverage_pct": round(100 * bi_kept_freq / bi_total, 2) if bi_total else 0,
        "trigram_coverage_pct": round(100 * tri_kept_freq / tri_total, 2) if tri_total else 0,
    }

    del bigram_counts, trigram_counts
    return bigrams, trigrams, stats


# ── Index assignment ─────────────────────────────────────────────────────────

def build_ngram_index(bigrams, trigrams):
    """
    Assign sequential indices to known N-grams.

    Layout for each table (bigram / trigram):
        [0]        → UNK embedding (for unknown N-grams)
        [1 ... N]  → known N-grams (exact, zero collisions)

    Returns:
        bigram_to_idx:   dict mapping (tok_a, tok_b) -> index
        trigram_to_idx:  dict mapping (tok_a, tok_b, tok_c) -> index
        bigram_table_size:  1 + len(bigrams)
        trigram_table_size: 1 + len(trigrams)
    """
    bigram_to_idx = {}
    for i, ng in enumerate(bigrams):
        bigram_to_idx[ng] = i + 1  # 1-indexed, 0 = UNK

    trigram_to_idx = {}
    for i, ng in enumerate(trigrams):
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
    # Convert tuple keys to strings for JSON serialization
    bigram_dict = {f"{a},{b}": idx for (a, b), idx in bigram_to_idx.items()}
    trigram_dict = {f"{a},{b},{c}": idx for (a, b, c), idx in trigram_to_idx.items()}

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
