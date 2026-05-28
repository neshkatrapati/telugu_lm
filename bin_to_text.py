#!/usr/bin/env python3
"""
Convert train.bin (uint32 token IDs) → text file with token strings.

Splits on EOS (id=3), skips BOS/PAD/UNK, writes one sentence per line
with space-separated token strings. Output is ready for count_5grams.py.

Usage:
    python bin_to_text.py --data ./train-data/train.bin --vocab ./tokenizer/vocab.txt --output train_tokens.txt
"""

import argparse
import numpy as np
from tqdm import tqdm


SPECIAL_IDS = {0, 1, 2, 3}  # pad, unk, bos, eos


def load_vocab(path: str) -> list[str]:
    """Load vocab.txt → list where index = token ID."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(description="Convert train.bin to text tokens")
    parser.add_argument("--data", required=True, help="Path to train.bin (uint32)")
    parser.add_argument("--vocab", required=True, help="Path to vocab.txt")
    parser.add_argument("--output", required=True, help="Output text file")
    parser.add_argument("--eos-id", type=int, default=3, help="EOS token ID (default: 3)")
    parser.add_argument("--min-length", type=int, default=5,
                        help="Skip sentences shorter than this many tokens (default: 5)")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    vocab_size = len(vocab)
    print(f"Vocab: {vocab_size:,} tokens from {args.vocab}")

    data = np.memmap(args.data, dtype=np.uint32, mode="r")
    n_tokens = len(data)
    print(f"Loaded {n_tokens:,} tokens from {args.data}")

    # Find EOS positions
    eos_positions = np.where(data == args.eos_id)[0]
    print(f"Found {len(eos_positions):,} EOS markers → ~{len(eos_positions):,} sentences")

    n_written = 0
    n_skipped = 0
    n_oov = 0

    with open(args.output, "w", encoding="utf-8") as f:
        start = 0
        for eos_pos in tqdm(eos_positions, desc="Sentences", unit="sent"):
            chunk = data[start:eos_pos]  # exclude EOS
            start = eos_pos + 1

            # Filter out special tokens
            tokens = []
            for tid in chunk:
                tid = int(tid)
                if tid in SPECIAL_IDS:
                    continue
                if tid >= vocab_size:
                    n_oov += 1
                    continue
                tokens.append(vocab[tid])

            if len(tokens) >= args.min_length:
                f.write(" ".join(tokens))
                f.write("\n")
                n_written += 1
            else:
                n_skipped += 1

        # Trailing tokens after last EOS
        if start < n_tokens:
            chunk = data[start:n_tokens]
            tokens = []
            for tid in chunk:
                tid = int(tid)
                if tid in SPECIAL_IDS:
                    continue
                if tid >= vocab_size:
                    n_oov += 1
                    continue
                tokens.append(vocab[tid])
            if len(tokens) >= args.min_length:
                f.write(" ".join(tokens))
                f.write("\n")
                n_written += 1

    print(f"\nDone:")
    print(f"  Written:  {n_written:,} sentences")
    print(f"  Skipped:  {n_skipped:,} (< {args.min_length} tokens)")
    print(f"  OOV IDs:  {n_oov:,}")
    print(f"  Output:   {args.output}")


if __name__ == "__main__":
    main()
