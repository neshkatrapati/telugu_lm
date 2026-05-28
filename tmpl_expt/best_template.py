#!/usr/bin/env python3
"""
Given a 5-gram "A B C D E", compute the count-scaled PMI for every
possible template (every subset of positions with at least 1 word
filled) and report the template with the highest scaled PMI.

Count-scaled PMI dampens the well-known PMI bias toward rare events:

    scaled_PMI = log2(1 + c(template)) * PMI

where PMI for a template with filled positions S ⊆ {0,1,2,3,4}, |S|=k:

    PMI = log2( c(template) * N^(k-1) / product_i(marginal_i) )

    c(template) = SUM of counts of all 5-grams matching the template
    marginal_i  = c(word_i at position i), summed over all wildcards
    N           = total count of all 5-grams

For k=1 (single word filled), PMI = 0 always → scaled_PMI = 0.
The interesting cases are k ≥ 2.

Usage:
    python best_template.py "A B C D E"
    python best_template.py A B C D E
"""

import sys
import math
import duckdb
from itertools import combinations

DB_PATH = "/Users/nesh/Documents/telugu_lm/tmpl_expt/5grams.duckdb"
COLS = ["w1", "w2", "w3", "w4", "w5"]


def get_template_count(db, words, filled_positions):
    """Get sum of counts for 5-grams matching the template."""
    clauses = [f"{COLS[i]} = ?" for i in filled_positions]
    params = [words[i] for i in filled_positions]
    sql = f"SELECT COALESCE(SUM(count), 0) FROM fivegrams WHERE {' AND '.join(clauses)}"
    return db.execute(sql, params).fetchone()[0]


def get_marginal(db, word, position):
    """Get c(word at position) = sum of counts where w_pos = word."""
    sql = f"SELECT COALESCE(SUM(count), 0) FROM fivegrams WHERE {COLS[position]} = ?"
    return db.execute(sql, [word]).fetchone()[0]


def compute_pmi(template_count, marginals, N, k):
    """
    PMI = log2( c(template) * N^(k-1) / prod(marginals) )

    k = number of filled positions
    """
    if template_count == 0:
        return float("-inf"), float("-inf")

    # Use log-space to avoid overflow
    log_pmi = (
        math.log2(template_count)
        + (k - 1) * math.log2(N)
        - sum(math.log2(m) for m in marginals)
    )

    # Count-scaled PMI: dampen rare-event bias
    scaled = math.log2(1 + template_count) * log_pmi

    return log_pmi, scaled


def format_template(words, filled_positions):
    """Pretty-print a template: filled words and _ for wildcards."""
    parts = []
    for i in range(5):
        if i in filled_positions:
            parts.append(words[i])
        else:
            parts.append("_")
    return " ".join(parts)


def main():
    # Parse input
    if len(sys.argv) == 2 and " " in sys.argv[1]:
        words = sys.argv[1].split()
    elif len(sys.argv) == 6:
        words = sys.argv[1:6]
    else:
        print("Usage: python best_template.py \"A B C D E\"")
        print("       python best_template.py A B C D E")
        sys.exit(1)

    if len(words) != 5:
        print(f"Error: expected 5 words, got {len(words)}: {words}")
        sys.exit(1)

    print(f"5-gram: {' '.join(words)}\n")

    db = duckdb.connect(DB_PATH, read_only=True)

    # Total count
    N = db.execute("SELECT SUM(count) FROM fivegrams").fetchone()[0]
    print(f"N (total 5-gram count) = {N:,}\n")

    # Pre-compute all 5 marginals (one per position)
    marginals = {}
    for i in range(5):
        m = get_marginal(db, words[i], i)
        marginals[(words[i], i)] = m
        print(f"  marginal c({format_template(words, {i})}) = {m:,}")
    print()

    # Enumerate all 2^5 - 1 = 31 non-empty subsets of positions
    best_scaled = float("-inf")
    best_template = None
    best_count = 0
    best_k = 0
    best_raw_pmi = 0

    results = []

    for k in range(1, 6):
        for positions in combinations(range(5), k):
            filled = set(positions)
            template_str = format_template(words, filled)

            # Template count
            c_template = get_template_count(db, words, positions)

            # Marginals for filled positions
            m_list = [marginals[(words[i], i)] for i in positions]

            # PMI (raw and count-scaled)
            pmi, scaled = compute_pmi(c_template, m_list, N, k)

            results.append((scaled, pmi, k, template_str, c_template, positions))

            if scaled > best_scaled:
                best_scaled = scaled
                best_raw_pmi = pmi
                best_template = template_str
                best_count = c_template
                best_k = k

    # Sort by scaled PMI descending
    results.sort(key=lambda x: -x[0])

    # Print all templates
    print(f"{'Template':<30} {'k':>2} {'count':>10} {'PMI':>12} {'scaled':>12}")
    print("-" * 72)
    for scaled, pmi, k, tpl, count, _ in results:
        pmi_str = f"{pmi:.4f}" if pmi > float("-inf") else "-inf"
        sc_str = f"{scaled:.2f}" if scaled > float("-inf") else "-inf"
        marker = " <<<" if tpl == best_template else ""
        print(f"{tpl:<30} {k:>2} {count:>10,} {pmi_str:>12} {sc_str:>12}{marker}")

    print(f"\n{'='*72}")
    print(f"BEST TEMPLATE: {best_template}")
    print(f"  scaled_PMI = {best_scaled:.2f},  PMI = {best_raw_pmi:.4f},  count = {best_count:,},  k = {best_k}")

    db.close()


if __name__ == "__main__":
    main()
