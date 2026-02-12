#!/usr/bin/env python3
"""
BPE Training for Non-Telugu Text
==================================
Extracts non-Telugu tokens from the Morfessor-segmented corpus and trains
a Byte Pair Encoding (BPE) model on them. The resulting subword vocabulary
uses the same @@ continuation convention as the Morfessor morphemes.

Usage:
    python train_bpe.py --seg-corpus ./data/morfessor/segmented_corpus/sangraha/ --num-merges 8000
    python train_bpe.py --seg-corpus ./path/to/telugu_verified.seg.txt --num-merges 4000

Output:
    bpe_merges.txt   — ordered merge rules (one per line: "a b")
    bpe_vocab.tsv    — subword + frequency
"""

import re
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TELUGU_CHAR = re.compile(r"[\u0C00-\u0C7F]")


# ---------------------------------------------------------------------------
# Step 1: Extract non-Telugu tokens from segmented corpus
# ---------------------------------------------------------------------------
def extract_non_telugu(seg_corpus_path: Path, separator: str = "@@") -> Counter:
    """Scan .seg.txt files and collect non-Telugu token frequencies.

    A token is non-Telugu if its base form (with @@ stripped) contains
    no Telugu characters (Unicode 0C00-0C7F).

    Returns Counter mapping raw surface form (no @@) -> frequency.
    """
    if seg_corpus_path.is_file():
        seg_files = [seg_corpus_path]
    else:
        seg_files = sorted(seg_corpus_path.rglob("*.seg.txt"))

    if not seg_files:
        logger.error("No .seg.txt files found in %s", seg_corpus_path)
        sys.exit(1)

    from tqdm import tqdm

    logger.info("Scanning %d file(s) for non-Telugu tokens...", len(seg_files))
    word_freq: Counter = Counter()
    total_tokens = 0
    non_telugu_tokens = 0

    for fpath in seg_files:
        logger.info("  Scanning %s", fpath.name)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=fpath.name, unit=" lines"):
                for token in line.split():
                    total_tokens += 1
                    # Strip @@ to get the base form
                    base = token.rstrip(separator) if token.endswith(separator) else token
                    # Check if it contains any Telugu characters
                    if not TELUGU_CHAR.search(base):
                        word_freq[base] += 1
                        non_telugu_tokens += 1

    logger.info("Total tokens scanned: %d", total_tokens)
    logger.info("Non-Telugu tokens: %d (%.1f%%)", non_telugu_tokens,
                100 * non_telugu_tokens / total_tokens if total_tokens else 0)
    logger.info("Unique non-Telugu word types: %d", len(word_freq))

    return word_freq


# ---------------------------------------------------------------------------
# Step 2: Train BPE
# ---------------------------------------------------------------------------
def train_bpe(word_freqs: Counter, num_merges: int) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Train BPE on word frequencies.

    Args:
        word_freqs: Counter mapping word -> frequency
        num_merges: number of BPE merge operations to learn

    Returns:
        merges: ordered list of (a, b) merge pairs
        vocab: dict mapping subword -> frequency
    """
    from tqdm import tqdm

    # Initialize: each word is a tuple of characters
    # word_splits maps tuple-of-chars -> frequency
    word_splits: dict[tuple[str, ...], int] = {}
    for word, freq in word_freqs.items():
        chars = tuple(word)
        if chars:
            word_splits[chars] = word_splits.get(chars, 0) + freq

    logger.info("BPE training: %d unique words, %d target merges", len(word_splits), num_merges)

    merges: list[tuple[str, str]] = []

    for merge_idx in tqdm(range(num_merges), desc="BPE merges", unit=" merges"):
        # Count all adjacent pairs
        pair_freq: Counter = Counter()
        for word_chars, freq in word_splits.items():
            for i in range(len(word_chars) - 1):
                pair_freq[(word_chars[i], word_chars[i + 1])] += freq

        if not pair_freq:
            logger.info("No more pairs to merge at step %d", merge_idx)
            break

        # Find most frequent pair
        best_pair = pair_freq.most_common(1)[0][0]
        best_freq = pair_freq[best_pair]

        if best_freq < 2:
            logger.info("Best pair frequency dropped to %d at step %d, stopping", best_freq, merge_idx)
            break

        merges.append(best_pair)

        # Merge the best pair in all words
        merged = best_pair[0] + best_pair[1]
        new_word_splits: dict[tuple[str, ...], int] = {}
        for word_chars, freq in word_splits.items():
            new_chars = []
            i = 0
            while i < len(word_chars):
                if (i < len(word_chars) - 1 and
                        word_chars[i] == best_pair[0] and
                        word_chars[i + 1] == best_pair[1]):
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(word_chars[i])
                    i += 1
            new_word_splits[tuple(new_chars)] = freq
        word_splits = new_word_splits

    logger.info("Learned %d BPE merges", len(merges))

    # Build final subword vocab with frequencies
    vocab: Counter = Counter()
    for word_chars, freq in word_splits.items():
        for subword in word_chars:
            vocab[subword] += freq

    logger.info("BPE vocabulary: %d subword types", len(vocab))

    return merges, dict(vocab.most_common())


# ---------------------------------------------------------------------------
# Step 3: BPE encode
# ---------------------------------------------------------------------------
def bpe_encode(word: str, merges: list[tuple[str, str]], separator: str = "@@") -> list[str]:
    """Encode a word into BPE subwords using the learned merge table.

    Returns subwords with @@ on non-final pieces:
        "international" -> ["inter@@", "nation@@", "al"]

    Args:
        word: the raw word to encode
        merges: ordered list of (a, b) merge pairs from training
        separator: continuation marker (default: @@)
    """
    if not word:
        return []

    # Start with individual characters
    chars = list(word)

    # Apply merges in order
    for a, b in merges:
        merged = a + b
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars

    # Add @@ to all subwords except the last (word-final)
    result = []
    for i, subword in enumerate(chars):
        if i < len(chars) - 1:
            result.append(subword + separator)
        else:
            result.append(subword)

    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_merges(merges: list[tuple[str, str]], path: Path):
    """Save merge rules to file (one per line: "a b")."""
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")
    logger.info("Saved %d merge rules to %s", len(merges), path)


def load_merges(path: Path) -> list[tuple[str, str]]:
    """Load merge rules from file."""
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    return merges


def save_vocab(vocab: dict[str, int], path: Path):
    """Save BPE vocabulary to TSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("subword\tfrequency\n")
        for subword, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write(f"{subword}\t{freq}\n")
    logger.info("Saved %d subwords to %s", len(vocab), path)


def load_vocab(path: Path) -> dict[str, int]:
    """Load BPE vocabulary from TSV file."""
    vocab = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    return vocab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train BPE on non-Telugu text from segmented corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --seg-corpus ./data/morfessor/segmented_corpus/sangraha/
  %(prog)s --seg-corpus ./path/to/file.seg.txt --num-merges 4000
  %(prog)s --seg-corpus ./data/ --num-merges 8000 --output ./data/bpe/
        """,
    )
    parser.add_argument(
        "--seg-corpus", type=str, required=True,
        help="Path to segmented corpus file or directory (.seg.txt files)",
    )
    parser.add_argument(
        "--num-merges", type=int, default=8000,
        help="Number of BPE merge operations (default: 8000)",
    )
    parser.add_argument(
        "--output", type=str, default="./data/morfessor/bpe",
        help="Output directory for BPE files (default: ./data/morfessor/bpe)",
    )
    parser.add_argument(
        "--separator", type=str, default="@@",
        help="Continuation marker (default: @@)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=5,
        help="Minimum word frequency to include in BPE training (default: 5)",
    )

    args = parser.parse_args()

    seg_path = Path(args.seg_corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract non-Telugu tokens
    word_freqs = extract_non_telugu(seg_path, args.separator)

    # Filter by minimum frequency
    if args.min_freq > 1:
        before = len(word_freqs)
        word_freqs = Counter({w: f for w, f in word_freqs.items() if f >= args.min_freq})
        logger.info("Filtered words by min_freq=%d: %d -> %d types", args.min_freq, before, len(word_freqs))

    if not word_freqs:
        logger.error("No non-Telugu tokens found. Nothing to train BPE on.")
        sys.exit(1)

    # Step 2: Train BPE
    merges, vocab = train_bpe(word_freqs, args.num_merges)

    # Step 3: Save outputs
    save_merges(merges, output_dir / "bpe_merges.txt")
    save_vocab(vocab, output_dir / "bpe_vocab.tsv")

    # Step 4: Test encoding
    logger.info("")
    logger.info("=" * 60)
    logger.info("BPE Test Encodings:")
    logger.info("=" * 60)
    test_words = ["international", "government", "2024", "https", "the", "COVID-19"]
    for word in test_words:
        encoded = bpe_encode(word, merges, args.separator)
        logger.info("  %-20s -> %s", word, " ".join(encoded))

    # Show top subwords
    logger.info("")
    logger.info("Top 30 BPE subwords:")
    for sw, freq in sorted(vocab.items(), key=lambda x: -x[1])[:30]:
        logger.info("  %-20s  %10d", sw, freq)

    logger.info("")
    logger.info("BPE training complete!")
    logger.info("  Merges:  %s", output_dir / "bpe_merges.txt")
    logger.info("  Vocab:   %s", output_dir / "bpe_vocab.tsv")


if __name__ == "__main__":
    main()
