#!/usr/bin/env python3
"""
For every 5-gram (with count >= threshold) in the DB, compute the
count-scaled PMI for all 31 template patterns and find the best.

Stores:
  1. Top K fully-instantiated 5-grams by scaled PMI (for exact lookup)
  2. Positional marginals (for on-the-fly PMI at query time)
  3. Total count N

Usage:
    python compress_5grams.py <input_db> <min_count> [--top-k K] [--out-db OUTPUT_DB]

    e.g. python compress_5grams.py 5grams.duckdb 10 --top-k 100000
"""

import sys
import os
import math
import json
import argparse
import duckdb
import time
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

COLS = ["w1", "w2", "w3", "w4", "w5"]

# Pre-generate all subsets of {0,1,2,3,4} with at least 2 filled positions
# (k=1 singletons excluded — too generic to be meaningful)
ALL_SUBSETS = []
for k in range(2, 6):
    for positions in combinations(range(5), k):
        ALL_SUBSETS.append(positions)


def build_marginals(db, min_count):
    """Pre-compute c(word at position) for all (word, position) pairs."""
    print("Building positional marginals...")
    marginals = {}
    for i, col in enumerate(COLS):
        print(f"  Position {i} ({col})...", end=" ", flush=True)
        rows = db.execute(
            f"SELECT {col}, SUM(count) FROM fivegrams "
            f"WHERE count >= {min_count} GROUP BY {col}"
        ).fetchall()
        for word, total in rows:
            marginals[(word, i)] = total
        print(f"{len(rows):,} distinct words")
    return marginals


def build_template_counts(db, min_count):
    """Pre-compute c(template) for every possible template pattern."""
    print("\nPre-computing template counts for all 31 patterns...")
    template_counts = {}
    for subset in tqdm(ALL_SUBSETS, desc="Patterns"):
        cols = [COLS[i] for i in subset]
        cols_str = ", ".join(cols)
        rows = db.execute(
            f"SELECT {cols_str}, SUM(count) as c FROM fivegrams "
            f"WHERE count >= {min_count} GROUP BY {cols_str}"
        ).fetchall()
        for row in rows:
            words_key = tuple(row[:-1])
            count = row[-1]
            template_counts[(subset, words_key)] = count
    return template_counts


def best_template_for_row(words, marginals, template_counts, N, log2N):
    """Find the template with highest count-scaled PMI for this 5-gram."""
    best_scaled = float("-inf")
    best_tpl = None

    for subset in ALL_SUBSETS:
        k = len(subset)
        filled_words = tuple(words[i] for i in subset)

        c_template = template_counts.get((subset, filled_words), 0)
        if c_template == 0:
            continue

        log_pmi = math.log2(c_template) + (k - 1) * log2N
        for i in subset:
            m = marginals.get((words[i], i), 0)
            if m == 0:
                log_pmi = float("-inf")
                break
            log_pmi -= math.log2(m)

        if log_pmi == float("-inf") or log_pmi < 0:
            continue

        scaled = math.log2(1 + c_template) * log_pmi

        if scaled > best_scaled:
            best_scaled = scaled
            best_tpl = (subset, filled_words)

    return best_tpl, best_scaled


def format_template_key(subset, filled_words):
    """Create a canonical string key for the template."""
    parts = []
    j = 0
    for i in range(5):
        if j < len(subset) and subset[j] == i:
            parts.append(filled_words[j])
            j += 1
        else:
            parts.append("_")
    return "\t".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Compress 5-grams: store top-K exact + marginals for on-the-fly PMI")
    parser.add_argument("input_db",
                        help="Path to input DuckDB with fivegrams table")
    parser.add_argument("min_count", type=int,
                        help="Only process 5-grams with count >= this")
    parser.add_argument("--top-k", type=int, default=100_000,
                        help="Keep top K fully-instantiated 5-grams by scaled PMI (default: 100000)")
    parser.add_argument("--out-db", default=None,
                        help="Output DuckDB path (default: templates.duckdb in same dir)")
    args = parser.parse_args()

    db_path = args.input_db
    min_count = args.min_count
    top_k = args.top_k

    if args.out_db:
        out_db_path = args.out_db
    else:
        out_db_path = os.path.join(os.path.dirname(db_path) or ".", "templates.duckdb")

    print(f"Input DB:    {db_path}")
    print(f"Output DB:   {out_db_path}")
    print(f"Min count:   {min_count}")
    print(f"Top K:       {top_k:,}\n")

    db = duckdb.connect(db_path, read_only=True)

    total_all = db.execute("SELECT COUNT(*) FROM fivegrams").fetchone()[0]
    total_types = db.execute(
        f"SELECT COUNT(*) FROM fivegrams WHERE count >= {min_count}"
    ).fetchone()[0]
    N = db.execute(
        f"SELECT SUM(count) FROM fivegrams WHERE count >= {min_count}"
    ).fetchone()[0]
    log2N = math.log2(N)

    print(f"Total 5-gram types in DB:       {total_all:>14,}")
    print(f"Types with count >= {min_count}:  {total_types:>14,}  "
          f"({total_types / total_all * 100:.1f}%)")
    print(f"Token mass (N):                 {N:>14,}\n")

    # Step 1: positional marginals
    marginals = build_marginals(db, min_count)
    print(f"  Total marginal entries: {len(marginals):,}\n")

    # Step 2: template counts for all 31 patterns
    t0 = time.time()
    template_counts = build_template_counts(db, min_count)
    print(f"  Total template count entries: {len(template_counts):,}")
    print(f"  Pre-computation took {time.time() - t0:.1f}s\n")

    # Step 3: compute scaled PMI for every qualifying 5-gram
    # Keep ALL for stats, but only store top-K exact in the DB
    print("Computing scaled PMI for each 5-gram...")
    t0 = time.time()

    # For each 5-gram: (w1..w5, count, best_subset, best_scaled_pmi)
    all_results = []  # (words, count, best_subset, best_scaled)
    pattern_hist = defaultdict(int)

    batch_size = 500_000
    offset = 0
    processed = 0

    pbar = tqdm(total=total_types, desc="5-grams", unit="ngram", unit_scale=True)

    while True:
        rows = db.execute(
            f"SELECT w1, w2, w3, w4, w5, count FROM fivegrams "
            f"WHERE count >= {min_count} "
            f"LIMIT {batch_size} OFFSET {offset}"
        ).fetchall()
        if not rows:
            break

        for row in rows:
            words = list(row[:5])
            count = row[5]

            best_tpl, best_scaled = best_template_for_row(
                words, marginals, template_counts, N, log2N
            )

            if best_tpl is None:
                subset_used = (0, 1, 2, 3, 4)
                best_scaled = 0.0
            else:
                subset_used = best_tpl[0]

            # Only keep fully-instantiated 5-grams for the top-K table
            if subset_used == (0, 1, 2, 3, 4):
                all_results.append((words, count, best_scaled))

            pattern_hist[subset_used] += 1

        processed += len(rows)
        pbar.update(len(rows))
        offset += batch_size

    pbar.close()
    elapsed = time.time() - t0
    print(f"  Processed {processed:,} 5-grams in {elapsed:.1f}s "
          f"({processed / elapsed:,.0f} ngrams/s)")
    print(f"  Fully-instantiated winners: {len(all_results):,}\n")

    # Sort by scaled PMI descending, take top K
    all_results.sort(key=lambda x: -x[2])
    top_results = all_results[:top_k]
    print(f"  Keeping top {len(top_results):,} by scaled PMI\n")

    # Close read-only connection
    db.close()

    # Step 4: write to output DB
    print(f"Writing results to {out_db_path}...")
    out_db = duckdb.connect(out_db_path)
    t_write = time.time()

    table_name = f"top5grams_min{min_count}"

    # --- Table 1: top-K exact 5-grams ---
    import pyarrow as pa
    data = {
        "w1": [r[0][0] for r in top_results],
        "w2": [r[0][1] for r in top_results],
        "w3": [r[0][2] for r in top_results],
        "w4": [r[0][3] for r in top_results],
        "w5": [r[0][4] for r in top_results],
        "count": [r[1] for r in top_results],
        "scaled_pmi": [round(r[2], 6) for r in top_results],
    }
    arrow_tbl = pa.table(data)
    out_db.execute(f"DROP TABLE IF EXISTS {table_name}")
    out_db.execute(f"CREATE TABLE {table_name} AS SELECT * FROM arrow_tbl")
    print(f"  Top-K table '{table_name}': {arrow_tbl.num_rows:,} rows")

    # --- Table 2: positional marginals ---
    marg_data = {
        "word": [k[0] for k in marginals],
        "position": [k[1] for k in marginals],
        "total_count": [v for v in marginals.values()],
    }
    arrow_marg = pa.table(marg_data)
    out_db.execute("DROP TABLE IF EXISTS marginals")
    out_db.execute("CREATE TABLE marginals AS SELECT * FROM arrow_marg")
    print(f"  Marginals table: {arrow_marg.num_rows:,} rows")

    # --- Table 3: metadata (N, min_count, log2N) ---
    out_db.execute("DROP TABLE IF EXISTS metadata")
    out_db.execute("""
        CREATE TABLE metadata (key VARCHAR, value DOUBLE)
    """)
    out_db.execute("INSERT INTO metadata VALUES (?, ?)", ["N", float(N)])
    out_db.execute("INSERT INTO metadata VALUES (?, ?)", ["min_count", float(min_count)])
    out_db.execute("INSERT INTO metadata VALUES (?, ?)", ["log2N", log2N])
    out_db.execute("INSERT INTO metadata VALUES (?, ?)", ["top_k", float(top_k)])
    print(f"  Metadata table: N={N:,}, min_count={min_count}, top_k={top_k:,}")

    print(f"  Done in {time.time() - t_write:.1f}s")

    # Stats
    print(f"\n{'='*60}")
    print(f"RESULTS  (min_count={min_count}, top_k={top_k:,})")
    print(f"{'='*60}")
    print(f"  Total qualifying 5-grams:       {total_types:>14,}")
    print(f"  Fully-instantiated winners:      {len(all_results):>14,}")
    print(f"  Stored in top-K table:           {len(top_results):>14,}")

    print(f"\n  Pattern distribution (by # filled positions):")
    k_hist = defaultdict(int)
    for subset, cnt in pattern_hist.items():
        k_hist[len(subset)] += cnt
    for k in sorted(k_hist):
        pct = k_hist[k] / total_types * 100
        print(f"    k={k} ({k} filled, {5-k} wild): {k_hist[k]:>12,}  ({pct:5.2f}%)")

    # Top 20
    print(f"\n  Top 20 stored 5-grams by scaled PMI:")
    top20 = out_db.execute(
        f"SELECT w1, w2, w3, w4, w5, count, scaled_pmi "
        f"FROM {table_name} ORDER BY scaled_pmi DESC LIMIT 20"
    ).fetchall()
    print(f"    {'5-gram':<50} {'count':>8} {'PMI':>10}")
    print(f"    {'-'*70}")
    for row in top20:
        ng = " ".join(row[:5])
        print(f"    {ng:<50} {row[5]:>8,} {row[6]:>10.2f}")

    out_db.close()
    print(f"\nResults saved to {out_db_path}")
    print(f"  Tables: {table_name}, marginals, metadata")


if __name__ == "__main__":
    main()
