#!/usr/bin/env python3
"""
Extend the morfessor_bpe_telugu_v4 tokenizer with retrieval special tokens.

Run once before Base v2 training:
    python extend_tokenizer.py --in tokenizer-new --out tokenizer-new-v5

Adds 9 tokens: <search>, </search>, <retrieved>, </retrieved>,
                <doc>, </doc>, <cite>, <think>, </think>

vocab grows 47822 -> 47831.
"""

import json
import argparse
import shutil
from pathlib import Path


RETRIEVAL_TOKENS = [
    "<search>",
    "</search>",
    "<retrieved>",
    "</retrieved>",
    "<doc>",
    "</doc>",
    "<cite>",
    "<think>",
    "</think>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True,
                        help="Input tokenizer dir (e.g., tokenizer-new)")
    parser.add_argument("--out", dest="out_dir", required=True,
                        help="Output tokenizer dir (will be created)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        raise SystemExit(f"input dir {in_dir} does not exist")
    if out_dir.exists():
        raise SystemExit(f"output dir {out_dir} already exists — refusing to overwrite")

    shutil.copytree(in_dir, out_dir)

    tok_path = out_dir / "tokenizer.json"
    with open(tok_path) as f:
        tok = json.load(f)

    assert tok.get("type") == "morfessor_bpe_telugu_v4", \
        f"unexpected tokenizer type: {tok.get('type')}"

    token_to_id = tok["token_to_id"]
    current_max = max(token_to_id.values())
    print(f"current vocab_size: {tok['vocab_size']}, max id: {current_max}")

    added = []
    next_id = current_max + 1
    for t in RETRIEVAL_TOKENS:
        if t in token_to_id:
            print(f"  already present: {t} -> {token_to_id[t]}")
            continue
        token_to_id[t] = next_id
        added.append((t, next_id))
        next_id += 1

    tok["vocab_size"] = max(token_to_id.values()) + 1
    if "special_tokens" not in tok:
        tok["special_tokens"] = {}
    for t, i in added:
        tok["special_tokens"][t] = i

    with open(tok_path, "w") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)

    vocab_txt = out_dir / "vocab.txt"
    if vocab_txt.exists():
        with open(vocab_txt, "a") as f:
            for t, _ in added:
                f.write(t + "\n")

    print(f"\nAdded {len(added)} tokens:")
    for t, i in added:
        print(f"  {i}: {t}")
    print(f"\nNew vocab_size: {tok['vocab_size']}")
    print(f"Output: {out_dir}")
    print("\nNext steps:")
    print("  1. Verify with your custom tokenizer loader that the new tokens encode as single IDs.")
    print(f"  2. Use --vocab-size {tok['vocab_size']} at training time (or let auto-detect handle it).")


if __name__ == "__main__":
    main()
