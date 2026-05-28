#!/usr/bin/env python3
"""
Mine high-frequency Roman Telugu word forms from codemix JSONL files.

Outputs:
  candidate_tokens.tsv   word<TAB>frequency<TAB>example contexts (top N)
  candidate_tokens.json  full data for further processing
  summary.txt            stats: total word forms, coverage, top-K histogram

Usage:
    python mine_codemix_tokens.py \\
        --inputs codemix-10k.jsonl codemix-5k-b2.jsonl codemix-5k-b2-partial-24w.jsonl \\
        --out-dir codemix-tokens \\
        --top-n 10000
"""
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# Strip punctuation (but keep word-internal hyphens / apostrophes)
_PUNCT_RE = re.compile(r"[^\w\-']+", flags=re.UNICODE)
_TELUGU_RE = re.compile(r"[ఀ-౿]")


def is_mostly_latin(word: str) -> bool:
    """True if >=80% of chars are Latin letters."""
    if not word:
        return False
    latin = sum(1 for c in word if c.isascii() and c.isalpha())
    return latin / len(word) >= 0.8 and latin > 0


def is_mostly_telugu(word: str) -> bool:
    """True if >=50% of chars are Telugu."""
    if not word:
        return False
    te = sum(1 for c in word if 0x0c00 <= ord(c) <= 0x0c7f)
    return te / len(word) >= 0.5


def normalize_word(word: str) -> str:
    """Lowercase, NFC-normalize."""
    word = unicodedata.normalize("NFC", word)
    return word.lower()


def tokenize_words(text: str):
    """Split into rough word tokens (whitespace + punctuation aware)."""
    if not text:
        return
    # First split on whitespace
    for tok in text.split():
        # Then split off leading/trailing punctuation (keep internal)
        # Use regex to extract word-character runs
        parts = _PUNCT_RE.split(tok)
        for p in parts:
            p = p.strip()
            if p:
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=10000,
                    help="Save top-N most frequent Roman tokens (will be filtered later)")
    ap.add_argument("--min-freq", type=int, default=5,
                    help="Skip word forms with fewer than this many occurrences")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Read all JSONL files, count word frequencies on each format
    roman_counts = Counter()       # codemix_roman + telugu_roman
    te_counts = Counter()          # codemix_te_en (Telugu words after substitution)
    n_records_ok = 0
    n_records_skip = 0

    print(f"[1/3] reading {len(args.inputs)} JSONL files", flush=True)
    for path in args.inputs:
        local_ok = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "_error" in r or "_skip" in r:
                    n_records_skip += 1
                    continue
                local_ok += 1
                # codemix_roman and telugu_roman: tokenize Roman words
                for fld in ("codemix_roman", "telugu_roman"):
                    text = r.get(fld) or ""
                    for tok in tokenize_words(text):
                        tok = normalize_word(tok)
                        if is_mostly_latin(tok):
                            roman_counts[tok] += 1
                # codemix_te_en: tokenize Telugu words (for context, not for adding)
                text_te = r.get("codemix_te_en") or ""
                for tok in tokenize_words(text_te):
                    tok = normalize_word(tok)
                    if is_mostly_telugu(tok):
                        te_counts[tok] += 1
        print(f"   {path.name}: {local_ok} ok records")
        n_records_ok += local_ok
    print(f"   total OK records: {n_records_ok:,}  (skipped {n_records_skip:,})", flush=True)

    print(f"[2/3] aggregating", flush=True)
    print(f"   unique Roman word forms: {len(roman_counts):,}")
    print(f"   total Roman word tokens (occurrences): {sum(roman_counts.values()):,}")
    print(f"   unique Telugu word forms (for ref): {len(te_counts):,}")

    # Filter by min_freq and sort
    candidates = [(w, c) for w, c in roman_counts.items() if c >= args.min_freq]
    candidates.sort(key=lambda x: -x[1])
    print(f"   passing min_freq>={args.min_freq}: {len(candidates):,}")

    # Take top N
    top = candidates[: args.top_n]
    print(f"   keeping top {len(top):,}", flush=True)

    # Save TSV
    tsv_path = args.out_dir / "candidate_tokens.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("rank\tword\tfreq\n")
        for i, (w, c) in enumerate(top, 1):
            f.write(f"{i}\t{w}\t{c}\n")
    print(f"   saved {tsv_path}", flush=True)

    # Save JSON
    json_path = args.out_dir / "candidate_tokens.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_records_ok": n_records_ok,
            "n_records_skip": n_records_skip,
            "n_unique_roman": len(roman_counts),
            "n_total_roman_occurrences": sum(roman_counts.values()),
            "n_after_min_freq": len(candidates),
            "min_freq": args.min_freq,
            "top_n": len(top),
            "candidates": [{"rank": i+1, "word": w, "freq": c} for i, (w, c) in enumerate(top)],
        }, f, ensure_ascii=False, indent=2)
    print(f"   saved {json_path}", flush=True)

    print(f"[3/3] coverage summary", flush=True)
    # How many TOKEN OCCURRENCES does the top-K cover?
    total_occurrences = sum(roman_counts.values())
    for k in [100, 500, 1000, 2000, 5000, 10000, len(candidates)]:
        if k > len(candidates): continue
        cov = sum(c for _, c in candidates[:k])
        print(f"   top-{k:>6}: covers {cov:>10,} / {total_occurrences:,}  ({100*cov/total_occurrences:.2f}%)")

    # Top 30 examples
    print(f"\n   TOP 30 Roman words:")
    for i, (w, c) in enumerate(top[:30], 1):
        print(f"     {i:>2}. {w:<25s} {c:>7,}")


if __name__ == "__main__":
    main()
