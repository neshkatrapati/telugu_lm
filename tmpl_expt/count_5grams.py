#!/usr/bin/env python3
"""
Count 5-grams directly from a text corpus and write to TSV + DuckDB.

Usage:
    python count_5grams.py <corpus_file> [--min-count N] [--out-dir DIR]

    e.g. python count_5grams.py ukwac_1M_san.txt --min-count 2 --out-dir output/

No KenLM needed. Adds <s> padding at sentence starts (like KenLM does).
"""

import sys
import os
import argparse
import time
from collections import Counter
from tqdm import tqdm


def count_lines(path):
    """Fast line count for progress bar."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Count 5-grams from corpus")
    parser.add_argument("corpus", help="Input text file (one sentence per line)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Only write 5-grams with count >= this (default: 1)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory (default: current dir)")
    parser.add_argument("--order", type=int, default=5,
                        help="N-gram order (default: 5)")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="Skip DuckDB output (just write TSV)")
    args = parser.parse_args()

    N = args.order
    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, f"{N}gram_counts.tsv")
    db_path = os.path.join(args.out_dir, f"{N}grams.duckdb")

    # Step 1: count lines for progress bar
    print(f"Counting lines in {args.corpus}...")
    total_lines = count_lines(args.corpus)
    print(f"  {total_lines:,} lines\n")

    # Step 2: count n-grams
    print(f"Counting {N}-grams...")
    counts = Counter()
    total_tokens = 0
    t0 = time.time()

    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Reading", unit="line",
                         unit_scale=True):
            tokens = line.strip().split()
            if not tokens:
                continue

            # Pad with <s> at start, </s> at end (like KenLM)
            padded = ["<s>"] * (N - 1) + tokens + ["</s>"]
            total_tokens += len(tokens)

            for i in range(len(padded) - N + 1):
                ngram = tuple(padded[i:i + N])
                counts[ngram] += 1

    elapsed = time.time() - t0
    print(f"\n  {total_tokens:,} tokens processed in {elapsed:.1f}s")
    print(f"  {len(counts):,} distinct {N}-gram types")

    # Filter by min count
    if args.min_count > 1:
        before = len(counts)
        counts = {ng: c for ng, c in counts.items() if c >= args.min_count}
        print(f"  After filtering (count >= {args.min_count}): "
              f"{len(counts):,} types (removed {before - len(counts):,})")

    # Step 3: write TSV
    print(f"\nWriting TSV to {tsv_path}...")
    t0 = time.time()
    header = "\t".join(f"w{i+1}" for i in range(N)) + "\tcount\n"

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for ngram, count in tqdm(
            sorted(counts.items(), key=lambda x: -x[1]),
            desc="Writing TSV", unit="ngram", unit_scale=True
        ):
            f.write("\t".join(ngram) + f"\t{count}\n")

    tsv_size = os.path.getsize(tsv_path) / 1e6
    print(f"  {tsv_size:.1f} MB written in {time.time() - t0:.1f}s")

    # Step 4: write DuckDB
    if not args.no_duckdb:
        try:
            import duckdb
        except ImportError:
            print("\n  duckdb not installed, skipping DB output.")
            print(f"  Install with: pip install duckdb")
            print(f"\nDone. TSV at: {tsv_path}")
            return

        print(f"\nLoading into DuckDB at {db_path}...")
        t0 = time.time()
        db = duckdb.connect(db_path)

        cols_def = ", ".join(f"w{i+1} VARCHAR" for i in range(N))
        db.execute(f"DROP TABLE IF EXISTS ngrams")
        db.execute(f"""
            CREATE TABLE ngrams AS
            SELECT * FROM read_csv(
                '{tsv_path}',
                delim='\t', header=true,
                columns={{ {', '.join(f"'w{i+1}': 'VARCHAR'" for i in range(N))}, 'count': 'BIGINT' }}
            )
        """)

        nrows = db.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
        db_size = os.path.getsize(db_path) / 1e6
        db.close()
        print(f"  {nrows:,} rows, {db_size:.1f} MB, took {time.time() - t0:.1f}s")

    print(f"\nDone!")
    print(f"  TSV:    {tsv_path}")
    if not args.no_duckdb:
        print(f"  DuckDB: {db_path}")


if __name__ == "__main__":
    main()
