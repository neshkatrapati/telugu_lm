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
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pack N-gram token IDs into single int64 keys for fast counting
# With vocab ~86K, MAX_TOKEN = 100000 is safe (86K^3 = 6.4e14 < 2^64)
MAX_TOKEN = 100_000


# ── Worker function for parallel counting ────────────────────────────────────

def _count_chunk(args):
    """
    Count bigrams and trigrams in a single chunk of the data.
    Runs in a worker process.

    Args: (data_path, start, end, is_last_chunk)
    Returns: (bigram_counter_dict, trigram_counter_dict, n_tokens_processed)
    """
    data_path, start, end, chunk_size = args

    # Each worker memmaps the file independently (no shared state)
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")

    # Read chunk with overlap for N-gram continuity at boundaries
    read_end = min(end + 2, len(data))
    chunk = np.array(data[start:read_end], dtype=np.int64)

    # How many tokens this chunk "owns" (not counting overlap)
    own_len = end - start

    # Bigrams: pairs within the owned region (plus one overlap token)
    bi_len = min(len(chunk) - 1, own_len)
    bi_keys = chunk[:bi_len] * MAX_TOKEN + chunk[1:bi_len + 1]
    unique_bi, counts_bi = np.unique(bi_keys, return_counts=True)
    bi_dict = dict(zip(unique_bi.tolist(), counts_bi.tolist()))

    # Trigrams: triples within the owned region (plus two overlap tokens)
    tri_len = min(len(chunk) - 2, own_len)
    tri_keys = (chunk[:tri_len] * (MAX_TOKEN * MAX_TOKEN)
                + chunk[1:tri_len + 1] * MAX_TOKEN
                + chunk[2:tri_len + 2])
    unique_tri, counts_tri = np.unique(tri_keys, return_counts=True)
    tri_dict = dict(zip(unique_tri.tolist(), counts_tri.tolist()))

    return bi_dict, tri_dict, own_len


# ── N-gram counting ─────────────────────────────────────────────────────────

def count_ngrams(data_path: Path, min_count: int, num_workers: int = 0):
    """
    Count bigram and trigram frequencies from a uint32 memmap file.
    Uses parallel workers for chunk processing + vectorized numpy within each chunk.

    Args:
        data_path:   path to train.bin
        min_count:   frequency cutoff
        num_workers:  0 = auto (cpu_count), 1 = single-process

    Returns:
        bigrams:  sorted list of (token_a, token_b) tuples that meet min_count
        trigrams: sorted list of (token_a, token_b, token_c) tuples that meet min_count
        stats:    dict with counting statistics
    """
    logger.info("Loading %s ...", data_path)
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    n_tokens = len(data)
    logger.info("  %d tokens (%.2f GB)", n_tokens, n_tokens * 4 / 1e9)

    if num_workers == 0:
        num_workers = min(cpu_count(), 8)  # cap at 8

    CHUNK = max(10_000_000, n_tokens // (num_workers * 4))  # at least 10M per chunk
    chunk_ranges = []
    for start in range(0, n_tokens, CHUNK):
        end = min(start + CHUNK, n_tokens)
        chunk_ranges.append((str(data_path), start, end, CHUNK))

    del data  # close memmap before forking

    logger.info("Counting N-grams: %d chunks × %d workers ...", len(chunk_ranges), num_workers)
    t0 = time.time()

    bigram_counts = Counter()
    trigram_counts = Counter()

    if num_workers == 1:
        # Single-process mode (useful for debugging)
        for chunk_args in tqdm(chunk_ranges, desc="Counting N-grams", unit="chunk"):
            bi_dict, tri_dict, _ = _count_chunk(chunk_args)
            bigram_counts.update(bi_dict)
            trigram_counts.update(tri_dict)
    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_count_chunk, args): i
                       for i, args in enumerate(chunk_ranges)}

            with tqdm(total=len(chunk_ranges), desc="Counting N-grams", unit="chunk") as pbar:
                for future in as_completed(futures):
                    bi_dict, tri_dict, _ = future.result()
                    bigram_counts.update(bi_dict)
                    trigram_counts.update(tri_dict)
                    pbar.update(1)

    elapsed = time.time() - t0
    logger.info("Counting done in %.1fs", elapsed)
    logger.info("  Unique bigrams:  %d", len(bigram_counts))
    logger.info("  Unique trigrams: %d", len(trigram_counts))

    # Decode keys back to tuples and apply cutoff
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
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: 0 = auto)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_bin = data_dir / "train.bin"

    if not train_bin.exists():
        logger.error("train.bin not found in %s", data_dir)
        logger.error("Run `python train_gpt.py prepare` first.")
        sys.exit(1)

    # ── Step 1: Count N-grams ──
    bigrams, trigrams, stats = count_ngrams(train_bin, args.min_count, args.workers)

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
