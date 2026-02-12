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
# Step 1: Extract non-Telugu tokens from segmented corpus (parallelized)
# ---------------------------------------------------------------------------

# Pre-compiled set of Telugu codepoints for fast membership test (no regex per token)
_TELUGU_CP_RANGE = range(0x0C00, 0x0C80)


def _has_telugu(s: str) -> bool:
    """Fast Telugu detection — checks codepoints directly, no regex."""
    for ch in s:
        if ord(ch) in _TELUGU_CP_RANGE:
            return True
    return False


def _scan_chunk(args: tuple) -> tuple[Counter, int, int]:
    """Worker: scan a chunk of lines, return (non_telugu_counter, total_toks, non_telugu_toks)."""
    lines, separator, sep_len = args
    freq: Counter = Counter()
    total = 0
    non_tel = 0
    for line in lines:
        for token in line.split():
            total += 1
            # Strip @@ suffix
            if token.endswith(separator):
                base = token[:-sep_len]
            else:
                base = token
            if not _has_telugu(base):
                freq[base] += 1
                non_tel += 1
    return freq, total, non_tel


def extract_non_telugu(seg_corpus_path: Path, separator: str = "@@", num_workers: int = 0) -> Counter:
    """Scan .seg.txt files and collect non-Telugu token frequencies.

    Parallelized: reads file in large chunks, distributes across workers.

    Returns Counter mapping raw surface form (no @@) -> frequency.
    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    if seg_corpus_path.is_file():
        seg_files = [seg_corpus_path]
    else:
        seg_files = sorted(seg_corpus_path.rglob("*.seg.txt"))

    if not seg_files:
        logger.error("No .seg.txt files found in %s", seg_corpus_path)
        sys.exit(1)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    CHUNK_SIZE = 50_000  # lines per chunk
    sep_len = len(separator)

    logger.info("Scanning %d file(s) for non-Telugu tokens (%d workers)...", len(seg_files), num_workers)
    word_freq: Counter = Counter()
    total_tokens = 0
    non_telugu_tokens = 0

    for fpath in seg_files:
        logger.info("  Scanning %s", fpath.name)

        # Read all lines and split into chunks
        chunks = []
        current_chunk = []
        line_count = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                current_chunk.append(line)
                line_count += 1
                if len(current_chunk) >= CHUNK_SIZE:
                    chunks.append((current_chunk, separator, sep_len))
                    current_chunk = []
        if current_chunk:
            chunks.append((current_chunk, separator, sep_len))

        logger.info("    %d lines -> %d chunks", line_count, len(chunks))

        # Process chunks in parallel
        if num_workers > 1 and len(chunks) > 1:
            with Pool(processes=min(num_workers, len(chunks))) as pool:
                for freq, total, non_tel in tqdm(
                    pool.imap_unordered(_scan_chunk, chunks),
                    total=len(chunks),
                    desc=fpath.name,
                    unit=" chunks",
                ):
                    word_freq += freq
                    total_tokens += total
                    non_telugu_tokens += non_tel
        else:
            # Single-threaded fallback
            for chunk_args in tqdm(chunks, desc=fpath.name, unit=" chunks"):
                freq, total, non_tel = _scan_chunk(chunk_args)
                word_freq += freq
                total_tokens += total
                non_telugu_tokens += non_tel

    logger.info("Total tokens scanned: %d", total_tokens)
    logger.info("Non-Telugu tokens: %d (%.1f%%)", non_telugu_tokens,
                100 * non_telugu_tokens / total_tokens if total_tokens else 0)
    logger.info("Unique non-Telugu word types: %d", len(word_freq))

    return word_freq


# ---------------------------------------------------------------------------
# Step 2: Train BPE
# ---------------------------------------------------------------------------
def train_bpe(word_freqs: Counter, num_merges: int) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Train BPE on word frequencies — fast incremental version.

    Instead of rescanning every word on every merge, maintains:
      - pair_counts: global pair frequency counter
      - pair_to_words: index mapping each pair -> set of word indices that contain it

    When a merge happens, only words containing that pair are updated,
    and pair counts are adjusted incrementally (subtract old pairs, add new ones).

    ~50-100x faster than naive rescan for typical vocab sizes.

    Args:
        word_freqs: Counter mapping word -> frequency
        num_merges: number of BPE merge operations to learn

    Returns:
        merges: ordered list of (a, b) merge pairs
        vocab: dict mapping subword -> frequency
    """
    from tqdm import tqdm
    import heapq

    # Initialize: each word stored as a mutable list of symbols
    # words[i] = (list_of_symbols, frequency)
    words: list[tuple[list[str], int]] = []
    for word, freq in word_freqs.items():
        chars = list(word)
        if chars:
            words.append((chars, freq))

    logger.info("BPE training: %d unique words, %d target merges", len(words), num_merges)

    # Build initial pair counts and pair->word index
    pair_counts: Counter = Counter()
    # pair_to_words: maps pair -> set of word indices that contain it
    pair_to_words: dict[tuple[str, str], set[int]] = {}

    for wi, (symbols, freq) in enumerate(words):
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(wi)

    merges: list[tuple[str, str]] = []

    # Use a max-heap for fast best-pair lookup
    # Heap entries: (-count, pair) — negative because heapq is min-heap
    # We use lazy deletion: check if count is still current before using
    heap = [(-count, pair) for pair, count in pair_counts.items() if count >= 2]
    heapq.heapify(heap)

    pbar = tqdm(total=num_merges, desc="BPE merges", unit=" merges")

    while len(merges) < num_merges and heap:
        # Pop best pair (lazy deletion: skip stale entries)
        while heap:
            neg_count, best_pair = heapq.heappop(heap)
            actual_count = pair_counts.get(best_pair, 0)
            if actual_count >= 2 and actual_count == -neg_count:
                break  # valid entry
        else:
            break  # heap exhausted

        best_freq = actual_count
        if best_freq < 2:
            break

        merges.append(best_pair)
        merged = best_pair[0] + best_pair[1]

        # Get all words containing this pair (copy the set since we'll modify it)
        affected = list(pair_to_words.get(best_pair, set()))

        for wi in affected:
            symbols, freq = words[wi]

            # Find all positions where best_pair occurs and apply merge
            # First: subtract old pairs from counts
            for i in range(len(symbols) - 1):
                p = (symbols[i], symbols[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    if p in pair_to_words:
                        pair_to_words[p].discard(wi)

            # Apply merge
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols) - 1 and
                        symbols[i] == best_pair[0] and
                        symbols[i + 1] == best_pair[1]):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            # Update in place
            words[wi] = (new_symbols, freq)

            # Add new pairs from updated word
            for i in range(len(new_symbols) - 1):
                p = (new_symbols[i], new_symbols[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(wi)
                # Push to heap if promising
                if pair_counts[p] >= 2:
                    heapq.heappush(heap, (-pair_counts[p], p))

        # Clean up the merged pair
        pair_counts.pop(best_pair, None)
        pair_to_words.pop(best_pair, None)

        pbar.update(1)
        if len(merges) % 500 == 0:
            pbar.set_postfix({"vocab": f"{len(set(s for syms, _ in words for s in syms))}",
                              "best_freq": best_freq})

    pbar.close()

    logger.info("Learned %d BPE merges", len(merges))

    # Build final subword vocab with frequencies
    vocab: Counter = Counter()
    for symbols, freq in words:
        for subword in symbols:
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
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers for corpus scan (default: auto = cpu_count - 1)",
    )

    args = parser.parse_args()

    seg_path = Path(args.seg_corpus)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract non-Telugu tokens
    word_freqs = extract_non_telugu(seg_path, args.separator, args.workers)

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
