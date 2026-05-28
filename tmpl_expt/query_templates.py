#!/usr/bin/env python3
"""
Given a templates DB + source 5grams DB and an input sentence, slide a
5-gram window and for each window:
  1. Try exact match in the top-K table
  2. If miss, compute all 31 template PMI scores on-the-fly and pick best

Every window always gets a match.

Usage:
    python query_templates.py <templates_db> <fivegrams_db> <table_name> "sentence"

    e.g. python query_templates.py templates.duckdb 5grams.duckdb top5grams_min10 "some sentence here"
"""

import sys
import math
import argparse
import time
import duckdb
from itertools import combinations
from collections import defaultdict

COLS = ["w1", "w2", "w3", "w4", "w5"]

# All subsets of {0,1,2,3,4} with at least 2 filled positions (26 patterns)
ALL_SUBSETS = []
for k in range(2, 6):
    for positions in combinations(range(5), k):
        ALL_SUBSETS.append(positions)

# On-the-fly only considers partial templates (k=2 to k=4).
# k=5 (fully instantiated) is reserved for exact top-K lookup only.
ONFLY_SUBSETS = [s for s in ALL_SUBSETS if len(s) < 5]


def load_exact_index(db, table_name):
    """Load top-K exact 5-grams into a dict for O(1) lookup."""
    rows = db.execute(
        f"SELECT w1, w2, w3, w4, w5, count, scaled_pmi FROM {table_name}"
    ).fetchall()
    index = {}
    for row in rows:
        key = tuple(row[:5])
        index[key] = (row[5], row[6])  # (count, scaled_pmi)
    return index


def load_marginals(db):
    """Load pre-computed marginals from the templates DB."""
    rows = db.execute("SELECT word, position, total_count FROM marginals").fetchall()
    marginals = {}
    for word, pos, total in rows:
        marginals[(word, pos)] = total
    return marginals


def load_metadata(db):
    """Load N, log2N, min_count from metadata table."""
    rows = db.execute("SELECT key, value FROM metadata").fetchall()
    meta = {r[0]: r[1] for r in rows}
    return meta


def compute_template_count(fivegrams_db, subset, words):
    """Query the 5grams DB for a specific template count.
    e.g. subset=(0,2,4), words=('A','C','E') → SELECT SUM(count) WHERE w1='A' AND w3='C' AND w5='E'
    """
    conditions = []
    params = []
    for idx, pos in enumerate(subset):
        conditions.append(f"{COLS[pos]} = ?")
        params.append(words[idx])
    where = " AND ".join(conditions)
    result = fivegrams_db.execute(
        f"SELECT SUM(count) FROM fivegrams WHERE {where}", params
    ).fetchone()[0]
    return result or 0


def compute_best_template(ngram, fivegrams_db, marginals, N, log2N):
    """Compute scaled PMI for all 31 templates and return the best.
    This is the on-the-fly version — queries the 5grams DB directly.
    """
    best_scaled = float("-inf")
    best_subset = None
    best_count = 0

    for subset in ONFLY_SUBSETS:
        k = len(subset)
        filled_words = tuple(ngram[i] for i in subset)

        # Get template count from DB
        c_template = compute_template_count(fivegrams_db, subset, filled_words)
        if c_template == 0:
            continue

        # PMI in log-space
        log_pmi = math.log2(c_template) + (k - 1) * log2N
        valid = True
        for i in subset:
            m = marginals.get((ngram[i], i), 0)
            if m == 0:
                valid = False
                break
            log_pmi -= math.log2(m)

        if not valid:
            continue

        # Skip negative PMI — anti-associations are meaningless
        if log_pmi < 0:
            continue

        # Count-scaled PMI
        scaled = math.log2(1 + c_template) * log_pmi

        if scaled > best_scaled:
            best_scaled = scaled
            best_subset = subset
            best_count = c_template

    return best_subset, best_count, best_scaled


def format_template(ngram, subset):
    """Format the template: filled positions show words, others show _."""
    parts = []
    for i in range(5):
        if i in subset:
            parts.append(ngram[i])
        else:
            parts.append("_")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Query: exact lookup → on-the-fly PMI fallback")
    parser.add_argument("templates_db", help="Path to templates DuckDB (from compress)")
    parser.add_argument("fivegrams_db", help="Path to source 5grams DuckDB")
    parser.add_argument("table_name", help="Top-K table name (e.g. top5grams_min10)")
    parser.add_argument("sentence", help="Input sentence")
    args = parser.parse_args()

    # Load templates DB
    tpl_db = duckdb.connect(args.templates_db, read_only=True)

    tables = [r[0] for r in tpl_db.execute("SHOW TABLES").fetchall()]
    if args.table_name not in tables:
        print(f"ERROR: table '{args.table_name}' not found. Available: {tables}")
        sys.exit(1)

    print("Loading data...", flush=True)
    t0 = time.time()

    exact_index = load_exact_index(tpl_db, args.table_name)
    marginals = load_marginals(tpl_db)
    meta = load_metadata(tpl_db)
    tpl_db.close()

    N = meta["N"]
    log2N = meta["log2N"]

    print(f"  Exact index: {len(exact_index):,} entries")
    print(f"  Marginals:   {len(marginals):,} entries")
    print(f"  N={N:,.0f}, log2N={log2N:.4f}")
    print(f"  Loaded in {time.time() - t0:.2f}s\n")

    # Open source 5grams DB for on-the-fly queries
    fg_db = duckdb.connect(args.fivegrams_db, read_only=True)

    tokens = args.sentence.strip().split()
    print(f"Input: {args.sentence}")
    print(f"Tokens: {len(tokens)}\n")

    if len(tokens) < 5:
        print("ERROR: need at least 5 tokens")
        sys.exit(1)

    # Header
    print(f"{'Pos':<5} {'5-gram':<50} {'Match':<7} "
          f"{'Template':<50} {'Count':>8} {'PMI':>10}")
    print("-" * 135)

    stats = {"exact": 0, "computed": 0}
    t0 = time.time()

    for i in range(len(tokens) - 4):
        ngram = tuple(tokens[i:i + 5])
        ngram_str = " ".join(ngram)

        # 1. Try exact match
        if ngram in exact_index:
            count, pmi = exact_index[ngram]
            tpl_str = ngram_str + " [full]"
            print(f"{i:<5} {ngram_str:<50} {'exact':<7} "
                  f"{tpl_str:<50} {count:>8,} {pmi:>10.2f}")
            stats["exact"] += 1
            continue

        # 2. On-the-fly PMI computation
        best_subset, best_count, best_scaled = compute_best_template(
            ngram, fg_db, marginals, N, log2N
        )

        if best_subset is not None:
            k = len(best_subset)
            wild = 5 - k
            tpl_str = format_template(ngram, best_subset)
            if wild > 0:
                label = f" [k={k}]"
            else:
                label = " [full]"
            print(f"{i:<5} {ngram_str:<50} {'onfly':<7} "
                  f"{tpl_str + label:<50} {best_count:>8,} {best_scaled:>10.2f}")
        else:
            print(f"{i:<5} {ngram_str:<50} {'onfly':<7} "
                  f"{'(no data)':<50} {'':>8} {'':>10}")

        stats["computed"] += 1

    fg_db.close()
    elapsed = time.time() - t0

    # Summary
    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"Summary: {total} windows ({elapsed:.2f}s, "
          f"{total/elapsed:.0f} windows/s)")
    print(f"  Exact (top-K hit): {stats['exact']:>6} ({stats['exact']/total*100:5.1f}%)")
    print(f"  On-the-fly PMI:    {stats['computed']:>6} ({stats['computed']/total*100:5.1f}%)")


if __name__ == "__main__":
    main()
