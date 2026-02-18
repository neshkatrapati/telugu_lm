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

Memory-efficient: each chunk is deduplicated to (key, count) pairs before writing
to disk. The merge step only processes unique N-grams per chunk, not raw
occurrences, so it's fast even for large corpora.

Usage:
    python build_ngram_vocab.py --data ./train_data --output ./tokenizer/ngram_vocab.json --min-count 50
"""

import os
import sys
import json
import time
import heapq
import tempfile
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

# With vocab ~86K, MAX_TOKEN = 100000 is safe (86K^3 = 6.4e14 < 2^63)
MAX_TOKEN = np.int64(100_000)


# ── Phase 1: Chunk, deduplicate, write to disk ──────────────────────────────

def _write_deduped_chunks(data_path: Path, n_tokens: int, order: int,
                          chunk_size: int, tmp_dir: str):
    """
    Stream through data in chunks. For each chunk:
      1. Build packed int64 N-gram keys
      2. np.unique → (unique_keys, counts) — deduplicates within chunk
      3. Write (key, count) pairs as int64 + int32 to temp file

    This is fast (numpy vectorized) and memory-bounded (one chunk at a time).
    The output files are MUCH smaller than raw keys since unique << raw.

    Returns: list of (tmp_path, n_unique_in_chunk) tuples, total raw N-grams
    """
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    chunk_files = []
    total_raw = 0
    total_unique_across_chunks = 0
    label = "bigram" if order == 2 else "trigram"
    overlap = order - 1

    with tqdm(total=n_tokens, desc=f"Counting {label}s", unit="tok", unit_scale=True) as pbar:
        for start in range(0, n_tokens - overlap, chunk_size):
            end = min(start + chunk_size + overlap, n_tokens)
            chunk = data[start:end].astype(np.int64)
            own_len = min(chunk_size, n_tokens - overlap - start)

            if own_len <= 0:
                pbar.update(end - start - overlap)
                break

            # Build packed keys
            if order == 2:
                keys = chunk[:own_len] * MAX_TOKEN + chunk[1:own_len + 1]
            else:
                keys = (chunk[:own_len] * (MAX_TOKEN * MAX_TOKEN)
                        + chunk[1:own_len + 1] * MAX_TOKEN
                        + chunk[2:own_len + 2])

            # Deduplicate within chunk
            unique_keys, counts = np.unique(keys, return_counts=True)
            counts = counts.astype(np.int64)

            # Interleave (key, count) pairs and write
            # Layout: [key0, count0, key1, count1, ...]
            pairs = np.empty(len(unique_keys) * 2, dtype=np.int64)
            pairs[0::2] = unique_keys
            pairs[1::2] = counts

            tmp_path = os.path.join(tmp_dir, f"{label}_{len(chunk_files)}.bin")
            pairs.tofile(tmp_path)
            chunk_files.append((tmp_path, len(unique_keys)))

            total_raw += own_len
            total_unique_across_chunks += len(unique_keys)

            del chunk, keys, unique_keys, counts, pairs
            pbar.update(own_len)

    logger.info("  %d chunks, %d total raw %ss, %d unique entries across chunks",
                len(chunk_files), total_raw, label, total_unique_across_chunks)
    return chunk_files, total_raw, total_unique_across_chunks


# ── Phase 2: K-way merge of deduped chunks ──────────────────────────────────

def _iter_chunk_file(path, n_unique):
    """Yield (key, count) pairs from a deduped chunk file, in sorted order."""
    data = np.fromfile(path, dtype=np.int64)
    keys = data[0::2]
    counts = data[1::2]
    for i in range(n_unique):
        yield (int(keys[i]), int(counts[i]))


def _merge_deduped_chunks(chunk_files, min_count, order, total_unique_entries):
    """
    K-way merge of sorted (key, count) chunk files.
    Sum counts for the same key across chunks. Apply min_count filter.

    Much faster than merging raw keys because we're iterating over
    unique-per-chunk entries, not raw occurrences.

    Returns: (filtered_ngrams_as_tuples, total_occurrences, kept_occurrences)
    """
    label = "bigram" if order == 2 else "trigram"

    if not chunk_files:
        return [], 0, 0

    # Create iterators for each chunk file
    iterators = []
    for path, n_unique in chunk_files:
        iterators.append(_iter_chunk_file(path, n_unique))

    # K-way merge on the key (first element of each tuple)
    merged = heapq.merge(*iterators, key=lambda x: x[0])

    results = []
    total_occ = 0
    kept_occ = 0

    current_key = None
    current_count = 0

    decode_bi = lambda k: (int(k // MAX_TOKEN), int(k % MAX_TOKEN))
    decode_tri = lambda k: (int(k // (MAX_TOKEN * MAX_TOKEN)),
                            int(k % (MAX_TOKEN * MAX_TOKEN) // MAX_TOKEN),
                            int(k % MAX_TOKEN))
    decode_fn = decode_bi if order == 2 else decode_tri

    with tqdm(total=total_unique_entries, desc=f"Merging {label}s", unit="entry", unit_scale=True) as pbar:
        for key, count in merged:
            pbar.update(1)
            if key == current_key:
                current_count += count
            else:
                # Flush previous
                if current_key is not None:
                    total_occ += current_count
                    if current_count >= min_count:
                        results.append(decode_fn(current_key))
                        kept_occ += current_count
                current_key = key
                current_count = count

        # Flush last
        if current_key is not None:
            total_occ += current_count
            if current_count >= min_count:
                results.append(decode_fn(current_key))
                kept_occ += current_count

    results.sort()
    return results, total_occ, kept_occ


# ── Full pipeline for one N-gram order ───────────────────────────────────────

def count_and_filter(data_path: Path, min_count: int, order: int,
                     chunk_size: int = 10_000_000, tmp_dir_base: str = None):
    """
    Count N-grams of given order, return filtered list.
    Uses disk-based sort+merge. RAM usage ≈ one chunk (~80MB) + merge iterators.
    """
    data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
    n_tokens = len(data)
    del data

    label = "bigram" if order == 2 else "trigram"
    logger.info("")
    logger.info("=" * 50)
    logger.info(" %ss (order=%d)", label.capitalize(), order)
    logger.info("=" * 50)

    with tempfile.TemporaryDirectory(prefix=f"ngram_{label}_", dir=tmp_dir_base) as tmp_dir:
        # Phase 1: chunk → dedupe → write sorted pairs to disk
        chunk_files, total_raw, total_unique = _write_deduped_chunks(
            data_path, n_tokens, order, chunk_size, tmp_dir
        )

        # Phase 2: k-way merge + count + filter
        ngrams, total_occ, kept_occ = _merge_deduped_chunks(
            chunk_files, min_count, order, total_unique
        )
        # Temp files cleaned up here

    logger.info("  Result: %d %ss (coverage: %.2f%%)",
                len(ngrams), label,
                100 * kept_occ / total_occ if total_occ else 0)

    return ngrams, total_occ, kept_occ


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
        bigram_to_idx[ng] = i + 1

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
    parser.add_argument("--chunk-size", type=int, default=10_000_000,
                        help="Tokens per chunk (default: 10M, lower = less RAM)")
    parser.add_argument("--tmp-dir", type=str, default=None,
                        help="Directory for temp files (default: system temp)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_bin = data_dir / "train.bin"

    if not train_bin.exists():
        logger.error("train.bin not found in %s", data_dir)
        logger.error("Run `python train_gpt.py prepare` first.")
        sys.exit(1)

    data = np.memmap(str(train_bin), dtype=np.uint32, mode="r")
    n_tokens = len(data)
    del data
    logger.info("Data: %s — %d tokens (%.2f GB)", train_bin, n_tokens, n_tokens * 4 / 1e9)

    t0 = time.time()

    # ── Bigrams first (temp files freed before trigrams start) ──
    bigrams, bi_total, bi_kept = count_and_filter(
        train_bin, args.min_count, order=2,
        chunk_size=args.chunk_size, tmp_dir_base=args.tmp_dir
    )

    # ── Trigrams ──
    trigrams, tri_total, tri_kept = count_and_filter(
        train_bin, args.min_count, order=3,
        chunk_size=args.chunk_size, tmp_dir_base=args.tmp_dir
    )

    elapsed = time.time() - t0
    logger.info("")
    logger.info("Total counting time: %.1fs", elapsed)

    if len(bigrams) == 0 and len(trigrams) == 0:
        logger.error("No N-grams survived the cutoff! Try lowering --min-count")
        sys.exit(1)

    # ── Assign indices ──
    bigram_to_idx, trigram_to_idx, bi_table_size, tri_table_size = build_ngram_index(
        bigrams, trigrams
    )

    # ── Save ──
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

        "stats": {
            "n_tokens": int(n_tokens),
            "bigram_occurrences_total": int(bi_total),
            "trigram_occurrences_total": int(tri_total),
            "bigrams_after_cutoff": len(bigrams),
            "trigrams_after_cutoff": len(trigrams),
            "bigram_coverage_pct": round(100 * bi_kept / bi_total, 2) if bi_total else 0,
            "trigram_coverage_pct": round(100 * tri_kept / tri_total, 2) if tri_total else 0,
        },
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
    logger.info("  Tokens:          %d", n_tokens)
    logger.info("  Min count:       %d", args.min_count)
    logger.info("  Bigrams:         %d (coverage: %.1f%%)",
                len(bigrams),
                100 * bi_kept / bi_total if bi_total else 0)
    logger.info("  Trigrams:        %d (coverage: %.1f%%)",
                len(trigrams),
                100 * tri_kept / tri_total if tri_total else 0)
    logger.info("  Bigram table:    %d rows (%d known + 1 UNK)",
                bi_table_size, len(bigrams))
    logger.info("  Trigram table:   %d rows (%d known + 1 UNK)",
                tri_table_size, len(trigrams))
    logger.info("")

    vocab_size = 86075
    d = 64
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
