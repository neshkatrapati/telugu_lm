#!/usr/bin/env python3
"""
Process an entire corpus through the template query pipeline.

Preloads ALL template counts + marginals into memory, then processes
the corpus with pure dict lookups using multiprocessing.

Output:
  pattern_ids.bin     — int32 array, one per 5-gram window (-1 = no match)
  pattern_vocab.jsonl — line N = pattern ID N
  pattern_stats.json  — summary statistics

Usage:
    python process_corpus_patterns.py \\
        --corpus train_tokens.txt \\
        --templates-db templates.duckdb \\
        --fivegrams-db 5grams.duckdb \\
        --output-dir ./pattern_data
"""

import os
import sys
import json
import math
import time
import argparse
import struct
import multiprocessing as mp
from itertools import combinations
from collections import defaultdict

import numpy as np
import duckdb
from tqdm import tqdm

COLS = ["w1", "w2", "w3", "w4", "w5"]

ALL_SUBSETS = []
for k in range(2, 6):
    for positions in combinations(range(5), k):
        ALL_SUBSETS.append(positions)

ONFLY_SUBSETS = [s for s in ALL_SUBSETS if len(s) < 5]

# Module-level globals for worker processes (set via initializer)
_exact_index = None
_template_counts = None
_marginals = None
_log2N = None
_min_pmi = None


def _worker_init(exact_index, template_counts, marginals, log2N, min_pmi):
    global _exact_index, _template_counts, _marginals, _log2N, _min_pmi
    _exact_index = exact_index
    _template_counts = template_counts
    _marginals = marginals
    _log2N = log2N
    _min_pmi = min_pmi


def _process_chunk(lines):
    """Process a list of lines. Returns per-window results + aggregate stats.
    per_sent_stats: list of (n_windows, n_exact, {k: count}) per sentence."""
    results = []
    n_exact = 0
    n_template = 0
    n_miss = 0
    n_short = 0
    # Per-k hit counts across all sentences in this chunk
    k_hits = {2: 0, 3: 0, 4: 0, 5: 0}
    # Per-sentence: (n_windows, n_exact, n_k2, n_k3, n_k4)
    per_sent = []

    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 5:
            n_short += 1
            continue

        n_windows = len(tokens) - 4
        s_exact = 0
        s_k = {2: 0, 3: 0, 4: 0}

        for i in range(n_windows):
            ngram = (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3], tokens[i+4])

            # 1. Exact
            hit = _exact_index.get(ngram)
            if hit is not None:
                count, pmi = hit
                if pmi >= _min_pmi:
                    results.append(((0, 1, 2, 3, 4), ngram, "exact", pmi, count))
                    n_exact += 1
                    k_hits[5] += 1
                    s_exact += 1
                else:
                    results.append(None)
                    n_miss += 1
                continue

            # 2. Template fallback
            if _template_counts:
                best_scaled = -1.0
                best_subset = None
                best_count = 0

                for subset in ONFLY_SUBSETS:
                    k = len(subset)
                    filled = tuple(ngram[j] for j in subset)

                    c_template = _template_counts.get((subset, filled), 0)
                    if c_template == 0:
                        continue

                    log_pmi = math.log2(c_template) + (k - 1) * _log2N
                    valid = True
                    for j in subset:
                        m = _marginals.get((ngram[j], j), 0)
                        if m == 0:
                            valid = False
                            break
                        log_pmi -= math.log2(m)

                    if not valid or log_pmi < 0:
                        continue

                    scaled = math.log2(1 + c_template) * log_pmi
                    if scaled > best_scaled:
                        best_scaled = scaled
                        best_subset = subset
                        best_count = c_template

                if best_subset is not None and best_scaled >= _min_pmi:
                    words = tuple(ngram[j] for j in best_subset)
                    bk = len(best_subset)
                    results.append((best_subset, words, "template", best_scaled, best_count))
                    n_template += 1
                    k_hits[bk] += 1
                    s_k[bk] += 1
                    continue

            results.append(None)
            n_miss += 1

        per_sent.append((n_windows, s_exact, s_k[2], s_k[3], s_k[4]))

    return results, n_exact, n_template, n_miss, n_short, k_hits, per_sent


# ---- Preloading ----

def load_exact_index(db, table_name):
    rows = db.execute(
        f"SELECT w1, w2, w3, w4, w5, count, scaled_pmi FROM {table_name}"
    ).fetchall()
    index = {}
    for row in rows:
        index[tuple(row[:5])] = (row[5], row[6])
    return index


def load_marginals(db):
    rows = db.execute("SELECT word, position, total_count FROM marginals").fetchall()
    return {(w, p): c for w, p, c in rows}


def load_metadata(db):
    rows = db.execute("SELECT key, value FROM metadata").fetchall()
    return {r[0]: r[1] for r in rows}


def build_template_counts(db, min_count):
    print("\nPreloading template counts for k=2..4 (25 patterns)...")
    template_counts = {}
    for subset in tqdm(ONFLY_SUBSETS, desc="  Patterns"):
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
    print(f"  Loaded {len(template_counts):,} template entries")
    return template_counts


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        buf = f.read(1 << 20)
        while buf:
            n += buf.count(b"\n")
            buf = f.read(1 << 20)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Process corpus → pattern_ids.bin (multiprocessing)")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--templates-db", required=True)
    parser.add_argument("--fivegrams-db", required=True)
    parser.add_argument("--table-name", default="top5grams_min10")
    parser.add_argument("--output-dir", default="./pattern_data")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-pmi", type=float, default=0.0)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (default: cpu_count)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Lines per worker chunk (default: 10000)")
    args = parser.parse_args()

    n_workers = args.workers or mp.cpu_count()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load ---
    print("Loading templates DB...", flush=True)
    t0 = time.time()
    tpl_db = duckdb.connect(args.templates_db, read_only=True)
    exact_index = load_exact_index(tpl_db, args.table_name)
    marginals = load_marginals(tpl_db)
    meta = load_metadata(tpl_db)
    tpl_db.close()

    N = meta["N"]
    log2N = meta["log2N"]
    print(f"  Exact index: {len(exact_index):,}")
    print(f"  Marginals:   {len(marginals):,}")
    print(f"  N={N:,.0f}, log2N={log2N:.4f}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    template_counts = {}
    if not args.exact_only:
        fg_db = duckdb.connect(args.fivegrams_db, read_only=True)
        t1 = time.time()
        template_counts = build_template_counts(fg_db, args.min_count)
        fg_db.close()
        print(f"  Preload took {time.time() - t1:.1f}s")
    else:
        print("  EXACT-ONLY mode")

    # --- Count lines ---
    print("\nCounting lines...", flush=True)
    n_lines = count_lines(args.corpus)
    print(f"  {n_lines:,} lines")
    print(f"  Using {n_workers} workers, chunk_size={args.chunk_size}")

    # --- Pattern vocab (built in main process from worker results) ---
    pattern_to_id = {}
    pattern_meta = []

    def get_or_create_pattern(positions, words, match_type, scaled_pmi, count):
        key = (positions, words)
        pid = pattern_to_id.get(key)
        if pid is not None:
            if scaled_pmi > pattern_meta[pid]["scaled_pmi"]:
                pattern_meta[pid]["scaled_pmi"] = scaled_pmi
            pattern_meta[pid]["hits"] += 1
            return pid
        pid = len(pattern_meta)
        pattern_to_id[key] = pid
        pattern_meta.append({
            "positions": list(positions),
            "words": list(words),
            "k": len(positions),
            "type": match_type,
            "scaled_pmi": scaled_pmi,
            "count": count,
            "hits": 1,
        })
        return pid

    # --- Process with multiprocessing ---
    out_bin_path = os.path.join(args.output_dir, "pattern_ids.bin")
    total_exact = 0
    total_template = 0
    total_miss = 0
    total_short = 0
    total_windows = 0
    total_k_hits = {2: 0, 3: 0, 4: 0, 5: 0}

    # Per-sentence accumulators for averages
    all_sent_windows = []    # n_windows per sentence
    all_sent_exact = []      # n_exact per sentence
    all_sent_k2 = []
    all_sent_k3 = []
    all_sent_k4 = []

    def line_chunks(f, chunk_size):
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    t_start = time.time()

    with open(out_bin_path, "wb") as out_f, \
         open(args.corpus, "r", encoding="utf-8") as corpus_f, \
         mp.Pool(n_workers, initializer=_worker_init,
                 initargs=(exact_index, template_counts, marginals, log2N, args.min_pmi)) as pool:

        chunks = line_chunks(corpus_f, args.chunk_size)
        pbar = tqdm(total=n_lines, desc="Processing", unit=" lines",
                    dynamic_ncols=True, smoothing=0.05)

        for results, n_e, n_t, n_m, n_s, k_hits, per_sent in pool.imap(_process_chunk, chunks):
            total_exact += n_e
            total_template += n_t
            total_miss += n_m
            total_short += n_s
            for k in k_hits:
                total_k_hits[k] += k_hits[k]

            # Collect per-sentence stats
            for (nw, se, sk2, sk3, sk4) in per_sent:
                all_sent_windows.append(nw)
                all_sent_exact.append(se)
                all_sent_k2.append(sk2)
                all_sent_k3.append(sk3)
                all_sent_k4.append(sk4)

            # Convert results to pattern IDs and write
            ids = []
            for r in results:
                if r is None:
                    ids.append(-1)
                else:
                    positions, words, match_type, scaled_pmi, count = r
                    pid = get_or_create_pattern(positions, words, match_type, scaled_pmi, count)
                    ids.append(pid)

            if ids:
                out_f.write(struct.pack(f"<{len(ids)}i", *ids))

            total_windows += len(results)
            pbar.update(args.chunk_size)
            pbar.set_postfix(
                exact=total_exact, tmpl=total_template,
                miss=total_miss, patterns=len(pattern_meta),
                refresh=False)

        pbar.close()

    elapsed = time.time() - t_start

    # --- Write pattern vocab ---
    vocab_path = os.path.join(args.output_dir, "pattern_vocab.jsonl")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for entry in pattern_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Per-sentence averages ---
    n_sents = len(all_sent_windows)
    sent_windows = np.array(all_sent_windows, dtype=np.float64)
    sent_exact = np.array(all_sent_exact, dtype=np.float64)
    sent_k2 = np.array(all_sent_k2, dtype=np.float64)
    sent_k3 = np.array(all_sent_k3, dtype=np.float64)
    sent_k4 = np.array(all_sent_k4, dtype=np.float64)
    sent_any_hit = sent_exact + sent_k2 + sent_k3 + sent_k4

    # Avoid div-by-zero for sentences with 0 windows (shouldn't happen but safe)
    safe_windows = np.where(sent_windows > 0, sent_windows, 1)
    sent_hit_rate = sent_any_hit / safe_windows

    per_sent_stats = {
        "avg_windows_per_sent": float(sent_windows.mean()),
        "avg_exact_per_sent": float(sent_exact.mean()),
        "avg_k2_per_sent": float(sent_k2.mean()),
        "avg_k3_per_sent": float(sent_k3.mean()),
        "avg_k4_per_sent": float(sent_k4.mean()),
        "avg_any_hit_per_sent": float(sent_any_hit.mean()),
        "avg_hit_rate_per_sent": float(sent_hit_rate.mean()),
        "median_hit_rate_per_sent": float(np.median(sent_hit_rate)),
    }

    # --- Stats ---
    total = total_exact + total_template + total_miss
    stats = {
        "total_windows": total,
        "total_sentences": n_sents,
        "exact_hits": total_exact,
        "template_hits": total_template,
        "no_match": total_miss,
        "short_lines": total_short,
        "unique_patterns": len(pattern_meta),
        "exact_pct_of_total": 100.0 * total_exact / max(total, 1),
        "template_pct_of_total": 100.0 * total_template / max(total, 1),
        "no_match_pct_of_total": 100.0 * total_miss / max(total, 1),
        "hit_pct_of_total": 100.0 * (total_exact + total_template) / max(total, 1),
        "hits_by_k": {str(k): v for k, v in sorted(total_k_hits.items())},
        "pct_by_k": {str(k): 100.0 * v / max(total, 1) for k, v in sorted(total_k_hits.items())},
        "per_sentence": per_sent_stats,
        "elapsed_sec": elapsed,
        "windows_per_sec": total / max(elapsed, 0.001),
    }

    pmis = [p["scaled_pmi"] for p in pattern_meta if p["scaled_pmi"] > 0]
    if pmis:
        pmis.sort()
        stats["pmi_min"] = pmis[0]
        stats["pmi_p25"] = pmis[len(pmis) // 4]
        stats["pmi_median"] = pmis[len(pmis) // 2]
        stats["pmi_p75"] = pmis[3 * len(pmis) // 4]
        stats["pmi_max"] = pmis[-1]

    stats_path = os.path.join(args.output_dir, "pattern_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"DONE — {total:,} windows in {elapsed:.1f}s ({total/elapsed:,.0f} windows/sec)")
    print(f"       {n_sents:,} sentences ({total_short:,} skipped < 5 tokens)")

    print(f"\n--- Global counts (% of total 5-gram windows) ---")
    print(f"  Exact (k=5):     {total_k_hits[5]:>12,}  ({100.0*total_k_hits[5]/max(total,1):5.1f}%)")
    print(f"  Template k=4:    {total_k_hits[4]:>12,}  ({100.0*total_k_hits[4]/max(total,1):5.1f}%)")
    print(f"  Template k=3:    {total_k_hits[3]:>12,}  ({100.0*total_k_hits[3]/max(total,1):5.1f}%)")
    print(f"  Template k=2:    {total_k_hits[2]:>12,}  ({100.0*total_k_hits[2]/max(total,1):5.1f}%)")
    total_hits = total_exact + total_template
    print(f"  ─────────────────────────────────────────")
    print(f"  Any hit:         {total_hits:>12,}  ({100.0*total_hits/max(total,1):5.1f}%)")
    print(f"  No match:        {total_miss:>12,}  ({100.0*total_miss/max(total,1):5.1f}%)")

    print(f"\n--- Per-sentence averages (across {n_sents:,} sentences) ---")
    ps = per_sent_stats
    print(f"  Avg windows/sent:  {ps['avg_windows_per_sent']:.1f}")
    print(f"  Avg exact/sent:    {ps['avg_exact_per_sent']:.2f}")
    print(f"  Avg k=4/sent:      {ps['avg_k4_per_sent']:.2f}")
    print(f"  Avg k=3/sent:      {ps['avg_k3_per_sent']:.2f}")
    print(f"  Avg k=2/sent:      {ps['avg_k2_per_sent']:.2f}")
    print(f"  Avg any hit/sent:  {ps['avg_any_hit_per_sent']:.2f}")
    print(f"  Avg hit rate/sent: {ps['avg_hit_rate_per_sent']*100:.1f}%")
    print(f"  Median hit rate:   {ps['median_hit_rate_per_sent']*100:.1f}%")

    print(f"\n--- Unique patterns: {len(pattern_meta):,} ---")
    if pmis:
        print(f"  PMI: min={stats['pmi_min']:.2f}  p25={stats['pmi_p25']:.2f}  "
              f"median={stats['pmi_median']:.2f}  p75={stats['pmi_p75']:.2f}  max={stats['pmi_max']:.2f}")

    print(f"\nOutput:")
    print(f"  {out_bin_path} ({os.path.getsize(out_bin_path) / 1e9:.2f} GB)")
    print(f"  {vocab_path} ({len(pattern_meta):,} patterns)")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
