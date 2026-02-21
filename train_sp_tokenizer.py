#!/usr/bin/env python3
"""
Train a SentencePiece Unigram tokenizer (48K vocab) for Telugu.

SentencePiece Unigram uses MDL-based pruning internally:
  - Starts with a large seed vocabulary (characters + frequent substrings)
  - Iteratively removes tokens whose removal least increases corpus loss
  - Stops at the target vocab_size

This produces a linguistically-aware 48K vocab without needing Morfessor.

Requirements:
  - Raw Telugu text corpus (one sentence per line, or one doc per line)
  - pip install sentencepiece

Usage:
    python train_sp_tokenizer.py \
        --input corpus.txt \
        --output ./sp_tokenizer \
        --vocab-size 48000

    # If your corpus is split across multiple files:
    python train_sp_tokenizer.py \
        --input-dir ./data/raw_text/ \
        --output ./sp_tokenizer \
        --vocab-size 48000

    # If you only have train.bin (token IDs), you need the original text.
    # train.bin cannot be used — SentencePiece needs raw text.
"""

import os
import sys
import glob
import argparse
import tempfile
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_sentencepiece(input_path, output_dir, vocab_size=48000,
                        character_coverage=0.9999, model_type="unigram",
                        num_threads=4, max_sentence_length=16384,
                        seed_sentencepiece_size=1_000_000,
                        input_sentence_size=0):
    """
    Train a SentencePiece model.

    Args:
        input_path: Path to raw text file (one sentence/doc per line)
        output_dir: Directory to save model + vocab
        vocab_size: Target vocabulary size
        character_coverage: Fraction of characters to cover (0.9999 for Telugu)
        model_type: "unigram" (recommended) or "bpe"
        num_threads: Training parallelism
        max_sentence_length: Max bytes per sentence (skip longer ones)
        seed_sentencepiece_size: Seed vocab size before pruning
        input_sentence_size: If >0, subsample this many sentences for training
    """
    import sentencepiece as spm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / "sp_telugu")

    logger.info("=" * 60)
    logger.info("SentencePiece Unigram Tokenizer Training")
    logger.info("=" * 60)
    logger.info("  Input:              %s", input_path)
    logger.info("  Output:             %s", output_dir)
    logger.info("  Vocab size:         %d", vocab_size)
    logger.info("  Model type:         %s", model_type)
    logger.info("  Character coverage: %.4f", character_coverage)
    logger.info("  Seed vocab size:    %d", seed_sentencepiece_size)

    # Count input lines for logging
    logger.info("  Counting input lines ...")
    n_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for _ in f:
            n_lines += 1
    logger.info("  Input lines:        %d", n_lines)

    # Define special tokens matching our existing convention
    # pad=0, unk=1, bos=2, eos=3
    train_args = dict(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        max_sentence_length=max_sentence_length,
        seed_sentencepiece_size=seed_sentencepiece_size,

        # Special tokens — match our existing IDs
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        # Normalization — minimal for Telugu
        # Don't lowercase (Telugu doesn't have case, but English mixed in might)
        # NFKC normalization is generally safe
        normalization_rule_name="nfkc",

        # Byte fallback — handle any character not in the vocab
        byte_fallback=True,

        # Split digits individually (useful for number handling)
        split_digits=True,

        # Treat whitespace as a special prefix (like SentencePiece default)
        # The ▁ (U+2581) prefix marks word boundaries
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,

        # Training data sampling (if corpus is very large)
        shuffle_input_sentence=True,
    )

    if input_sentence_size > 0:
        train_args["input_sentence_size"] = input_sentence_size
        logger.info("  Subsampling:        %d sentences", input_sentence_size)

    logger.info("")
    logger.info("Training ...")
    spm.SentencePieceTrainer.train(**train_args)

    # Verify
    model_path = model_prefix + ".model"
    vocab_path = model_prefix + ".vocab"
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(vocab_path), f"Vocab not found: {vocab_path}"

    model_size = os.path.getsize(model_path)
    logger.info("")
    logger.info("Training complete!")
    logger.info("  Model: %s (%.1f MB)", model_path, model_size / 1e6)
    logger.info("  Vocab: %s", vocab_path)

    # Load and verify
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    actual_vocab = sp.get_piece_size()
    logger.info("")
    logger.info("Verification:")
    logger.info("  Actual vocab size: %d", actual_vocab)
    logger.info("  pad=%d unk=%d bos=%d eos=%d",
                sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())

    # Test encode/decode
    test_sentences = [
        "తెలుగు భాష చాలా అందమైన భాష",
        "హైదరాబాద్ తెలంగాణ రాజధాని",
        "Hello world this is a test 123",
        "విద్యార్థులు పరీక్షలకు సిద్ధమవుతున్నారు",
    ]

    logger.info("")
    logger.info("Sample encodings:")
    for sent in test_sentences:
        pieces = sp.encode(sent, out_type=str)
        ids = sp.encode(sent, out_type=int)
        decoded = sp.decode(ids)
        logger.info("  Input:   %s", sent)
        logger.info("  Pieces:  %s", " ".join(pieces))
        logger.info("  IDs:     %s", ids[:20])
        logger.info("  Decoded: %s", decoded)
        logger.info("  Match:   %s", "✓" if decoded == sent else "✗")
        logger.info("")

    # Count Telugu vs non-Telugu tokens in vocab
    n_telugu = 0
    n_byte_fallback = 0
    n_other = 0
    for i in range(actual_vocab):
        piece = sp.id_to_piece(i)
        if any(0x0C00 <= ord(c) <= 0x0C7F for c in piece):
            n_telugu += 1
        elif piece.startswith("<0x"):
            n_byte_fallback += 1
        else:
            n_other += 1

    logger.info("Vocab breakdown:")
    logger.info("  Telugu pieces:       %d", n_telugu)
    logger.info("  Byte fallback:       %d", n_byte_fallback)
    logger.info("  Other (English etc): %d", n_other)
    logger.info("  Special tokens:      4")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done. Model saved to: %s", output_dir)
    logger.info("=" * 60)

    return model_path


def merge_text_files(input_dir, output_path, extensions=(".txt", ".seg.txt")):
    """Merge all text files in a directory into one file for SentencePiece."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, f"**/*{ext}"), recursive=True))
    files.sort()

    if not files:
        logger.error("No text files found in %s with extensions %s", input_dir, extensions)
        sys.exit(1)

    logger.info("Merging %d files from %s → %s", len(files), input_dir, output_path)
    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        total_lines += 1
    logger.info("  Wrote %d lines", total_lines)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train SentencePiece Unigram tokenizer (48K) for Telugu"
    )

    # Input — either a single file or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str,
                             help="Path to raw text corpus (one sentence per line)")
    input_group.add_argument("--input-dir", type=str,
                             help="Directory of .txt files to merge")

    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for SP model")
    parser.add_argument("--vocab-size", type=int, default=48000,
                        help="Target vocabulary size (default: 48000)")
    parser.add_argument("--model-type", type=str, default="unigram",
                        choices=["unigram", "bpe"],
                        help="Model type (default: unigram)")
    parser.add_argument("--character-coverage", type=float, default=0.9999,
                        help="Character coverage (default: 0.9999)")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Training threads (default: 4)")
    parser.add_argument("--input-sentence-size", type=int, default=0,
                        help="Subsample N sentences for training (0=all)")
    args = parser.parse_args()

    # Handle input
    if args.input_dir:
        # Merge files into one temp file
        tmp_path = os.path.join(args.output, "merged_corpus.txt")
        os.makedirs(args.output, exist_ok=True)
        input_path = merge_text_files(args.input_dir, tmp_path)
    else:
        input_path = args.input

    if not os.path.exists(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    file_size = os.path.getsize(input_path)
    logger.info("Input corpus: %s (%.1f GB)", input_path, file_size / 1e9)

    train_sentencepiece(
        input_path=input_path,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentence_size,
    )


if __name__ == "__main__":
    main()
