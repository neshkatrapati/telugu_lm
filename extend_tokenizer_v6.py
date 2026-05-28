#!/usr/bin/env python3
"""
Extend tokenizer-new-v5 with top-K Roman Telugu word forms → tokenizer-new-v6.

Reads filtered_tokens.json (output of filter_tokens.py), picks the top N
candidates, and appends them as new vocabulary entries.

Usage:
    python extend_tokenizer_v6.py \\
        --in-tok tokenizer-new-v5 \\
        --candidates codemix-tokens/filtered_tokens.json \\
        --out-tok tokenizer-new-v6 \\
        --top-n 5000
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-tok", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out-tok", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5000)
    args = ap.parse_args()

    if args.out_tok.exists():
        raise SystemExit(f"output dir {args.out_tok} already exists")
    shutil.copytree(args.in_tok, args.out_tok)
    print(f"copied {args.in_tok} → {args.out_tok}")

    # Load existing tokenizer
    tok_path = args.out_tok / "tokenizer.json"
    with open(tok_path) as f:
        tok = json.load(f)
    assert tok["type"] == "morfessor_bpe_telugu_v4", f"unexpected type: {tok['type']}"
    token_to_id = tok["token_to_id"]
    current_max = max(token_to_id.values())
    print(f"current vocab_size: {tok['vocab_size']}, max id: {current_max}")

    # Load candidates
    candidates_data = json.load(open(args.candidates))
    candidates = candidates_data["candidates"]
    print(f"loaded {len(candidates):,} candidates from {args.candidates}")

    # Add top-N — skip any that are already in vocab
    next_id = current_max + 1
    added = []
    skipped_already_present = []
    for c in candidates[: args.top_n]:
        word = c["word"]
        if word in token_to_id:
            skipped_already_present.append(word)
            continue
        token_to_id[word] = next_id
        added.append({"id": next_id, "word": word, "freq": c["freq"],
                       "current_tokens": c["current_tokens"]})
        next_id += 1

    new_vocab_size = next_id  # since IDs are 0-indexed, vocab_size = max_id + 1
    tok["vocab_size"] = new_vocab_size

    # Save
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)
    print(f"saved {tok_path}  (new vocab_size: {new_vocab_size:,})")

    # Append to vocab.txt
    vocab_txt = args.out_tok / "vocab.txt"
    if vocab_txt.exists():
        with open(vocab_txt, "a", encoding="utf-8") as f:
            for a in added:
                f.write(a["word"] + "\n")

    # Save added-tokens record
    rec_path = args.out_tok / "added_tokens_v6.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump({
            "from_tokenizer": str(args.in_tok),
            "n_added": len(added),
            "n_skipped_already_present": len(skipped_already_present),
            "old_vocab_size": current_max + 1,
            "new_vocab_size": new_vocab_size,
            "added": added,
        }, f, ensure_ascii=False, indent=2)
    print(f"saved {rec_path}")

    print(f"\nAdded {len(added):,} new tokens; skipped {len(skipped_already_present)} already in vocab")
    print(f"vocab: {current_max+1:,} → {new_vocab_size:,}")
    print(f"\nFirst 20 added:")
    for a in added[:20]:
        print(f"  {a['id']:>6}  {a['word']:<30s} freq={a['freq']:>7}  (was {a['current_tokens']} tokens)")


if __name__ == "__main__":
    main()
