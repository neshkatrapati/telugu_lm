#!/usr/bin/env python3
"""
Entropy-based N-gram Filtering for Engram
==========================================
Reads a KenLM ARPA file and selects the most informative N-grams.

Score = P(ngram_joint) × surprisal
      ∝ count(ngram) × (-log10 P(last | context))

High score = frequent AND hard to predict = worth memorizing.
Low score  = predictable or rare = the transformer handles these.

Usage:
    python filter_ngrams_entropy.py --arpa ngrams.arpa --output ngram_vocab.json \
        --max-bigrams 200000 --max-trigrams 100000
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arpa(arpa_path):
    """
    Parse KenLM ARPA file. Returns:
        unigrams: {token_id_int: (log10_prob, log10_backoff)}
        bigrams:  [(a, b, log10_prob, backoff_or_0)]
        trigrams: [(a, b, c, log10_prob)]
    """
    unigrams = {}
    bigrams = []
    trigrams = []
    section = None

    logger.info("Parsing ARPA: %s", arpa_path)
    with open(arpa_path, "r") as f:
        for line in tqdm(f, desc="Parsing ARPA", unit=" lines", mininterval=1.0):
            line = line.strip()
            if not line:
                continue
            if line == "\\data\\":
                section = "header"
                continue
            if line == "\\end\\":
                break

            if section == "header":
                if line.startswith("ngram "):
                    order, count = line.split("=")
                    logger.info("  %s=%s", order.strip(), count.strip())
                continue

            if line.startswith("\\") and line.endswith(":"):
                order_str = line.strip("\\").strip(":")
                section = int(order_str.split("-")[0])
                logger.info("  Reading %d-grams ...", section)
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            log_prob = float(parts[0])
            tokens = parts[1].split()

            if section == 1 and len(tokens) == 1:
                tok = tokens[0]
                backoff = float(parts[2]) if len(parts) > 2 else 0.0
                if tok in ("<s>", "</s>", "<unk>"):
                    unigrams[tok] = (log_prob, backoff)
                else:
                    try:
                        unigrams[int(tok)] = (log_prob, backoff)
                    except ValueError:
                        pass

            elif section == 2 and len(tokens) == 2:
                try:
                    a, b = int(tokens[0]), int(tokens[1])
                    backoff = float(parts[2]) if len(parts) > 2 else 0.0
                    bigrams.append((a, b, log_prob, backoff))
                except ValueError:
                    pass

            elif section == 3 and len(tokens) == 3:
                try:
                    a, b, c = int(tokens[0]), int(tokens[1]), int(tokens[2])
                    trigrams.append((a, b, c, log_prob))
                except ValueError:
                    pass

    logger.info("Parsed: %d unigrams, %d bigrams, %d trigrams",
                len(unigrams), len(bigrams), len(trigrams))
    return unigrams, bigrams, trigrams


def score_bigrams(bigrams, unigrams):
    """
    Score = P(a) × P(b|a) × surprisal(b|a)
          = P(a,b) × (-log10 P(b|a))
    P(a) from unigrams, P(b|a) from bigram log_prob (ARPA gives conditional).
    """
    scored = []
    for a, b, log_p_b_given_a, _ in tqdm(bigrams, desc="Scoring bigrams", mininterval=1.0):
        if a not in unigrams:
            continue
        surprisal = -log_p_b_given_a
        if surprisal <= 0:
            continue
        log_p_a = unigrams[a][0]
        log_p_joint = log_p_a + log_p_b_given_a
        try:
            score = (10.0 ** log_p_joint) * surprisal
        except (OverflowError, ValueError):
            continue
        if score > 0:
            scored.append((a, b, score))
    logger.info("  Scored %d bigrams", len(scored))
    return scored


def score_trigrams(trigrams, unigrams, bigram_lookup):
    """
    Score = P(a,b,c) × surprisal(c|a,b)
          = P(a) × P(b|a) × P(c|a,b) × (-log10 P(c|a,b))

    We get P(a) from unigrams, P(b|a) from bigram_lookup, P(c|a,b) from ARPA.
    """
    scored = []
    for a, b, c, log_p_c_given_ab in tqdm(trigrams, desc="Scoring trigrams", mininterval=1.0):
        if a not in unigrams:
            continue
        surprisal = -log_p_c_given_ab
        if surprisal <= 0:
            continue

        log_p_a = unigrams[a][0]

        # Get P(b|a) from bigram lookup
        log_p_b_given_a = bigram_lookup.get((a, b))
        if log_p_b_given_a is None:
            # Backoff: P(b|a) ≈ backoff(a) × P(b)
            if b in unigrams:
                backoff_a = unigrams[a][1]  # log10 backoff weight
                log_p_b_given_a = backoff_a + unigrams[b][0]
            else:
                continue

        log_p_joint = log_p_a + log_p_b_given_a + log_p_c_given_ab
        try:
            score = (10.0 ** log_p_joint) * surprisal
        except (OverflowError, ValueError):
            continue
        if score > 0:
            scored.append((a, b, c, score))
    logger.info("  Scored %d trigrams", len(scored))
    return scored


def select_top_k(scored, k, label):
    """Sort by score descending, return top-K."""
    score_idx = 2 if label == "bigram" else 3
    scored.sort(key=lambda x: x[score_idx], reverse=True)
    selected = scored[:k]
    if selected:
        logger.info("  %s: top-%d from %d (score range: %.6f — %.6f)",
                    label, len(selected), len(scored),
                    selected[-1][score_idx], selected[0][score_idx])
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-based N-gram filtering for Engram"
    )
    parser.add_argument("--arpa", required=True, help="KenLM ARPA file")
    parser.add_argument("--output", required=True, help="Output ngram_vocab.json")
    parser.add_argument("--max-bigrams", type=int, default=200_000,
                        help="Top-K bigrams to keep (default: 200000)")
    parser.add_argument("--max-trigrams", type=int, default=100_000,
                        help="Top-K trigrams to keep (default: 100000)")
    parser.add_argument("--dump-scores", type=str, default=None,
                        help="Dump all scores to TSV for analysis")
    args = parser.parse_args()

    t0 = time.time()

    # ── Parse ──
    unigrams, bigrams, trigrams = parse_arpa(args.arpa)

    # ── Build bigram lookup for trigram scoring ──
    logger.info("Building bigram probability lookup ...")
    bigram_lookup = {}
    for a, b, log_prob, backoff in bigrams:
        bigram_lookup[(a, b)] = log_prob

    # ── Score ──
    logger.info("")
    bi_scored = score_bigrams(bigrams, unigrams)
    tri_scored = score_trigrams(trigrams, unigrams, bigram_lookup)
    del bigrams, trigrams, bigram_lookup

    # ── Dump (optional) ──
    if args.dump_scores:
        logger.info("Dumping scores to %s ...", args.dump_scores)
        with open(args.dump_scores, "w") as f:
            f.write("order\ttok_a\ttok_b\ttok_c\tscore\n")
            for a, b, s in bi_scored:
                f.write(f"2\t{a}\t{b}\t\t{s:.8f}\n")
            for a, b, c, s in tri_scored:
                f.write(f"3\t{a}\t{b}\t{c}\t{s:.8f}\n")

    # ── Select top-K ──
    logger.info("")
    sel_bi = select_top_k(bi_scored, args.max_bigrams, "bigram")
    sel_tri = select_top_k(tri_scored, args.max_trigrams, "trigram")
    del bi_scored, tri_scored

    # ── Sort by token IDs for deterministic output ──
    sel_bi.sort(key=lambda x: (x[0], x[1]))
    sel_tri.sort(key=lambda x: (x[0], x[1], x[2]))

    # ── Assign indices: [0]=UNK, [1..N]=known ──
    bigram_dict = {}
    for i, (a, b, _) in enumerate(sel_bi):
        bigram_dict[f"{a},{b}"] = i + 1

    trigram_dict = {}
    for i, (a, b, c, _) in enumerate(sel_tri):
        trigram_dict[f"{a},{b},{c}"] = i + 1

    bi_table = 1 + len(sel_bi)
    tri_table = 1 + len(sel_tri)

    # ── Save ──
    output = {
        "version": 3,
        "method": "entropy_mdl",
        "bigram_count": len(sel_bi),
        "bigram_table_size": bi_table,
        "trigram_count": len(sel_tri),
        "trigram_table_size": tri_table,
        "bigrams": bigram_dict,
        "trigrams": trigram_dict,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)

    elapsed = time.time() - t0
    file_mb = os.path.getsize(out_path) / 1e6

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("N-gram Vocab Summary (entropy-filtered)")
    logger.info("=" * 60)
    logger.info("  Bigrams:       %d (table: %d rows)", len(sel_bi), bi_table)
    logger.info("  Trigrams:      %d (table: %d rows)", len(sel_tri), tri_table)
    logger.info("  Output:        %s (%.1f MB)", out_path, file_mb)
    logger.info("  Time:          %.1fs", elapsed)

    d = 64
    vocab_size = 86075
    total_params = (vocab_size + bi_table + tri_table) * d
    logger.info("  Engram params (d=%d): %.1fM", d, total_params / 1e6)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
