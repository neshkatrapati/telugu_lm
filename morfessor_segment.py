#!/usr/bin/env python3
"""
Telugu Morphological Segmentation with Morfessor Baseline
==========================================================
Trains a Morfessor Baseline model on Telugu text and applies segmentation.

Pipeline:
  1. Extract words + frequencies from downloaded corpus
  2. Train Morfessor Baseline model
  3. Inspect segmentation quality on sample words
  4. Apply segmentation to full corpus

Requirements:
    pip install morfessor tqdm

Usage:
    # Full pipeline: train + segment
    python morfessor_segment.py --input ./data

    # Train only (inspect before segmenting everything)
    python morfessor_segment.py --input ./data --train-only

    # Segment only (using a previously trained model)
    python morfessor_segment.py --input ./data --segment-only --model ./data/morfessor/morfessor_telugu.bin

    # Custom sample size for training
    python morfessor_segment.py --input ./data --sample-size 10000000

    # Custom corpus weight (higher = less segmentation, lower = more)
    python morfessor_segment.py --input ./data --corpus-weight 1.0
"""

import os
import sys
import re
import argparse
import logging
import time
import json
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import morfessor  # noqa: F401
    except ImportError:
        missing.append("morfessor")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")

    if missing:
        logger.error(
            "Missing required packages: %s\n"
            "Install them with:\n"
            "    pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Telugu text utilities
# ---------------------------------------------------------------------------

# Telugu Unicode range: 0C00-0C7F
TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


def is_telugu(text: str) -> bool:
    """Check if text contains Telugu characters."""
    return bool(TELUGU_WORD_RE.search(text))


def extract_telugu_words(text: str) -> list[str]:
    """Extract Telugu words from a line of text."""
    return TELUGU_WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Step 1: Build word frequency list from corpus
# ---------------------------------------------------------------------------
def build_word_frequencies(
    input_dir: Path,
    sample_size: int,
    output_dir: Path,
) -> Path:
    """
    Scan downloaded data files and build a word frequency list.
    Stops after collecting `sample_size` word tokens.
    """
    from tqdm import tqdm

    freq_path = output_dir / "word_frequencies.txt"

    # Check if already built
    if freq_path.exists() and freq_path.stat().st_size > 1024:
        logger.info("Word frequency file already exists: %s", freq_path)
        logger.info("Delete it to rebuild. Loading existing file.")
        word_count = sum(1 for _ in open(freq_path, encoding="utf-8"))
        logger.info("Contains %d unique words", word_count)
        return freq_path

    logger.info("Building word frequency list (sample_size=%d tokens)...", sample_size)

    word_freq = Counter()
    total_tokens = 0
    files_processed = 0

    # Find all data files — input_dir can be a file or a directory
    if input_dir.is_file():
        data_files = [input_dir]
    else:
        data_files = []
        for ext in ("*.parquet", "*.jsonl", "*.txt"):
            data_files.extend(input_dir.rglob(ext))
        # Exclude files in the morfessor output directory itself
        data_files = [f for f in data_files if "morfessor" not in str(f)]
        data_files.sort(key=lambda f: f.stat().st_size, reverse=True)

    if not data_files:
        logger.error("No data files found in %s", input_dir)
        logger.error("Expected .parquet, .jsonl, or .txt files. Did you run the downloader first?")
        sys.exit(1)

    logger.info("Found %d data files to scan", len(data_files))

    for fpath in data_files:
        if total_tokens >= sample_size:
            break

        logger.info("Reading %s ...", fpath.name)
        files_processed += 1

        try:
            lines = _iter_text_from_file(fpath)
            for text in tqdm(lines, desc=fpath.name, unit=" docs", leave=False):
                words = extract_telugu_words(text)
                word_freq.update(words)
                total_tokens += len(words)

                if total_tokens >= sample_size:
                    break
        except Exception as e:
            logger.warning("Error reading %s: %s — skipping", fpath, e)
            continue

    logger.info(
        "Collected %d total tokens, %d unique words from %d files",
        total_tokens, len(word_freq), files_processed,
    )

    # Save word frequencies
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(freq_path, "w", encoding="utf-8") as f:
        for word, count in word_freq.most_common():
            f.write(f"{count} {word}\n")

    logger.info("Saved word frequencies to %s", freq_path)
    return freq_path


def _iter_text_from_file(fpath: Path):
    """Yield text strings from a data file (supports parquet, jsonl, txt)."""
    suffix = fpath.suffix.lower()

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            # Read in batches to avoid loading entire file into RAM
            parquet_file = pq.ParquetFile(fpath)
            for batch in parquet_file.iter_batches(
                batch_size=5000, columns=["text"]
            ):
                for text in batch.column("text").to_pylist():
                    if text:
                        yield text
        except ImportError:
            # Fallback: use pandas in chunks
            import pandas as pd
            for chunk in pd.read_parquet(fpath, columns=["text"], chunksize=5000):
                for text in chunk["text"].dropna():
                    yield text

    elif suffix == ".jsonl":
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    elif suffix == ".txt":
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


# ---------------------------------------------------------------------------
# Step 2: Train Morfessor model
# ---------------------------------------------------------------------------
def train_morfessor(
    freq_path: Path,
    output_dir: Path,
    corpus_weight: float,
    dampening: str,
) -> Path:
    """Train a Morfessor Baseline model on the word frequency list."""
    import morfessor

    model_path = output_dir / "morfessor_telugu.bin"

    # Check if model already exists
    if model_path.exists() and model_path.stat().st_size > 1024:
        logger.info("Morfessor model already exists: %s", model_path)
        logger.info("Delete it to retrain.")
        return model_path

    logger.info("Training Morfessor Baseline model...")
    logger.info("  Frequency file: %s", freq_path)
    logger.info("  Corpus weight:  %s", corpus_weight)
    logger.info("  Dampening:      %s", dampening)

    # Read word frequencies as (count, word) tuples
    # Our format is: "count word" per line
    # We parse it manually because read_corpus_file treats input as raw text
    # and ignores the count column, while read_corpus_list_file expects the
    # correct "count word" format but returns (count, (word,)) tuples.
    io = morfessor.MorfessorIO()
    word_counts = list(io.read_corpus_list_file(str(freq_path)))
    logger.info("Read %d word types from frequency file", len(word_counts))

    # Initialize model
    model = morfessor.BaselineModel(corpusweight=corpus_weight)

    # Set dampening via count_modifier
    import math
    if dampening == "log":
        count_modifier = lambda x: int(round(math.log(x + 1, 2)))
    elif dampening == "ones":
        count_modifier = lambda x: 1
    else:
        count_modifier = None

    # Load data
    model.load_data(word_counts, freqthreshold=2, count_modifier=count_modifier)

    # Train
    start = time.time()
    model.train_batch()
    elapsed = time.time() - start

    logger.info("Training completed in %.1f seconds", elapsed)

    # Save model
    io.write_binary_model_file(str(model_path), model)
    logger.info("Saved model to %s", model_path)

    return model_path


# ---------------------------------------------------------------------------
# Step 3: Inspect segmentation quality
# ---------------------------------------------------------------------------

# Common Telugu words to test segmentation on
SAMPLE_TELUGU_WORDS = [
    "విద్యార్థులకు",      # students + for
    "ప్రభుత్వంలో",        # government + in
    "అధ్యాపకులు",        # teachers (plural)
    "తెలుగువారి",        # Telugu people's
    "విశ్వవిద్యాలయం",    # university
    "సంస్కృతిని",        # culture (accusative)
    "వ్యవసాయదారులు",     # farmers (plural)
    "రాజకీయంగా",        # politically
    "అభివృద్ధిచెందుతున్న",  # developing
    "సమాచారాన్ని",       # information (accusative)
    "పరిశోధనలు",        # researches
    "ప్రజాస్వామ్యం",      # democracy
    "వేడుకలను",          # celebrations (accusative)
    "అనుభవాలు",         # experiences
    "నిర్వహించడానికి",    # in order to manage
    "చరిత్రకారులు",      # historians
    "ప్రపంచవ్యాప్తంగా",   # worldwide
    "సాంకేతికపరంగా",     # technologically
    "భాషాశాస్త్రవేత్తలు",  # linguists
    "అంతర్జాతీయంగా",     # internationally
]


def inspect_segmentation(model_path: Path, output_dir: Path):
    """Load model and show segmentation of sample words."""
    import morfessor

    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))

    sample_path = output_dir / "segmented_sample.txt"

    logger.info("")
    logger.info("=" * 70)
    logger.info("SAMPLE SEGMENTATIONS")
    logger.info("=" * 70)

    lines = []
    for word in SAMPLE_TELUGU_WORDS:
        segments = model.viterbi_segment(word)[0]
        segmented = " + ".join(segments)
        line = f"  {word:30s} -> {segmented}"
        logger.info(line)
        lines.append(f"{word}\t{' '.join(segments)}")

    # Also segment some words from the frequency list
    freq_path = output_dir / "word_frequencies.txt"
    if freq_path.exists():
        logger.info("")
        logger.info("Top frequent words:")
        logger.info("-" * 70)
        with open(freq_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 30:
                    break
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    count, word = parts
                    segments = model.viterbi_segment(word)[0]
                    segmented = " + ".join(segments)
                    log_line = f"  [{count:>8s}x] {word:30s} -> {segmented}"
                    logger.info(log_line)
                    lines.append(f"{word}\t{' '.join(segments)}\t{count}")

    logger.info("=" * 70)

    # Save sample
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved sample segmentations to %s", sample_path)


# ---------------------------------------------------------------------------
# Step 3b: Vocab statistics
# ---------------------------------------------------------------------------
def compute_vocab_stats(model_path: Path, freq_path: Path, output_dir: Path):
    """Compute and display morpheme vocabulary statistics from the trained model."""
    import morfessor

    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))

    # Collect morpheme stats by segmenting all words in the frequency file
    morpheme_freq = Counter()
    word_type_count = 0
    total_tokens = 0
    total_morpheme_tokens = 0
    unsegmented_count = 0

    logger.info("Computing vocabulary statistics...")

    with open(freq_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            count_str, word = parts
            try:
                count = int(count_str)
            except ValueError:
                continue

            word_type_count += 1
            total_tokens += count

            segments = model.viterbi_segment(word)[0]
            total_morpheme_tokens += len(segments) * count

            if len(segments) == 1:
                unsegmented_count += 1

            for seg in segments:
                morpheme_freq[seg] += count

    morpheme_types = len(morpheme_freq)
    avg_morphemes_per_word = total_morpheme_tokens / total_tokens if total_tokens > 0 else 0
    compression_ratio = word_type_count / morpheme_types if morpheme_types > 0 else 0

    # Morpheme length distribution
    lengths = [len(m) for m in morpheme_freq]
    avg_morpheme_len = sum(lengths) / len(lengths) if lengths else 0

    # Print stats
    logger.info("")
    logger.info("=" * 70)
    logger.info("VOCABULARY STATISTICS")
    logger.info("=" * 70)
    logger.info("  Word types (surface forms):      %d", word_type_count)
    logger.info("  Morpheme types (unique):         %d", morpheme_types)
    logger.info("  Compression ratio:               %.1fx (word types / morpheme types)", compression_ratio)
    logger.info("  Unsegmented words (kept intact):  %d (%.1f%%)",
                unsegmented_count, 100 * unsegmented_count / word_type_count if word_type_count else 0)
    logger.info("  Avg morphemes per word token:    %.2f", avg_morphemes_per_word)
    logger.info("  Avg morpheme length (chars):     %.1f", avg_morpheme_len)
    logger.info("  Total word tokens:               %d", total_tokens)
    logger.info("  Total morpheme tokens:           %d", total_morpheme_tokens)
    logger.info("-" * 70)

    # Top morphemes
    logger.info("  Top 30 most frequent morphemes:")
    for morph, cnt in morpheme_freq.most_common(30):
        logger.info("    %-20s  %10d", morph, cnt)

    # Bottom — rarest morphemes
    logger.info("")
    logger.info("  30 rarest morphemes:")
    for morph, cnt in morpheme_freq.most_common()[-30:]:
        logger.info("    %-20s  %10d", morph, cnt)

    logger.info("=" * 70)

    # Save full morpheme vocabulary
    vocab_path = output_dir / "morpheme_vocab.tsv"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("morpheme\tfrequency\n")
        for morph, cnt in morpheme_freq.most_common():
            f.write(f"{morph}\t{cnt}\n")
    logger.info("Saved full morpheme vocabulary (%d entries) to %s", morpheme_types, vocab_path)

    # Save summary stats
    stats_path = output_dir / "vocab_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"word_types: {word_type_count}\n")
        f.write(f"morpheme_types: {morpheme_types}\n")
        f.write(f"compression_ratio: {compression_ratio:.2f}\n")
        f.write(f"unsegmented_words: {unsegmented_count}\n")
        f.write(f"unsegmented_pct: {100 * unsegmented_count / word_type_count if word_type_count else 0:.1f}\n")
        f.write(f"avg_morphemes_per_word_token: {avg_morphemes_per_word:.2f}\n")
        f.write(f"avg_morpheme_length_chars: {avg_morpheme_len:.1f}\n")
        f.write(f"total_word_tokens: {total_tokens}\n")
        f.write(f"total_morpheme_tokens: {total_morpheme_tokens}\n")
    logger.info("Saved vocab stats to %s", stats_path)


# ---------------------------------------------------------------------------
# Step 4: Segment full corpus (with caching + multiprocessing)
# ---------------------------------------------------------------------------

def _build_segmentation_cache(model, freq_path: Path, separator: str) -> dict[str, str]:
    """
    Pre-segment all known words into a lookup dict.
    This avoids calling viterbi_segment millions of times during corpus pass.
    """
    from tqdm import tqdm

    cache = {}

    # Count lines first for progress bar
    num_lines = 0
    with open(freq_path, "r", encoding="utf-8") as f:
        for _ in f:
            num_lines += 1

    logger.info("Building segmentation cache from %d word types...", num_lines)

    with open(freq_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Caching segmentations", unit=" words"):
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            _, word = parts

            segments = model.viterbi_segment(word)[0]
            if len(segments) > 1:
                segmented_parts = []
                for i, seg in enumerate(segments):
                    if i < len(segments) - 1:
                        segmented_parts.append(seg + separator)
                    else:
                        segmented_parts.append(seg)
                cache[word] = " ".join(segmented_parts)
            else:
                cache[word] = word

    logger.info("Cached segmentations for %d word types", len(cache))
    return cache


def _segment_text_cached(cache: dict, model, text: str, separator: str) -> str:
    """
    Segment Telugu words in text using cache for known words.
    Falls back to viterbi_segment for unknown words (cache miss).
    """
    tokens = text.split()
    result = []

    for token in tokens:
        # No Telugu characters — keep as-is
        if not TELUGU_WORD_RE.search(token):
            result.append(token)
            continue

        # Purely Telugu token — use cache
        if TELUGU_WORD_RE.fullmatch(token):
            cached = cache.get(token)
            if cached is not None:
                result.append(cached)
            else:
                # Cache miss — segment and cache for future
                segments = model.viterbi_segment(token)[0]
                if len(segments) > 1:
                    parts = []
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            parts.append(seg + separator)
                        else:
                            parts.append(seg)
                    segmented = " ".join(parts)
                else:
                    segmented = token
                cache[token] = segmented
                result.append(segmented)
        else:
            # Mixed token — split on Telugu boundaries
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            for part in parts:
                if not part:
                    continue
                if TELUGU_WORD_RE.fullmatch(part):
                    cached = cache.get(part)
                    if cached is not None:
                        result.append(cached)
                    else:
                        segments = model.viterbi_segment(part)[0]
                        if len(segments) > 1:
                            seg_parts = []
                            for i, seg in enumerate(segments):
                                if i < len(segments) - 1:
                                    seg_parts.append(seg + separator)
                                else:
                                    seg_parts.append(seg)
                            segmented = " ".join(seg_parts)
                        else:
                            segmented = part
                        cache[part] = segmented
                        result.append(segmented)
                else:
                    result.append(part)

    return " ".join(result)


# Module-level globals for shared state across forked workers
_shared_cache = None
_shared_model = None
_shared_separator = None


def _init_worker(cache, model, separator):
    """Initializer for pool workers — sets shared globals from parent."""
    global _shared_cache, _shared_model, _shared_separator
    _shared_cache = cache
    _shared_model = model
    _shared_separator = separator


def _segment_batch(texts):
    """
    Worker function: segment a batch of texts using shared cache + model.
    Returns list of segmented texts. Skips docs that cause errors.
    """
    cache = _shared_cache  # read-only from parent via fork (copy-on-write)
    model = _shared_model
    separator = _shared_separator
    results = []
    for text in texts:
        try:
            results.append(_segment_text_cached(cache, model, text, separator))
        except Exception:
            # Skip problematic docs rather than hanging
            results.append(text)
    return results


def _segment_single(text):
    """Worker function: segment a single text. For imap_unordered."""
    try:
        return _segment_text_cached(_shared_cache, _shared_model, text, _shared_separator)
    except Exception:
        return text


BATCH_SIZE = 2000  # docs per batch sent to workers (smaller = more responsive)


def segment_corpus(
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
    separator: str,
    num_workers: int = 0,
):
    """
    Apply Morfessor segmentation to all corpus files.
    Builds cache ONCE, then processes each file with parallel batch segmentation.
    Progress bar updates per batch so you always see movement.
    """
    import morfessor
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    seg_dir = output_dir / "segmented_corpus"
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Look for word_frequencies.txt in output_dir first, then next to the model
    freq_path = output_dir / "word_frequencies.txt"
    if not freq_path.exists():
        freq_path = model_path.parent / "word_frequencies.txt"

    # Find all data files — input_dir can be a file or a directory
    if input_dir.is_file():
        data_files = [input_dir]
        base_dir = input_dir.parent
    else:
        data_files = []
        for ext in ("*.parquet", "*.jsonl", "*.txt"):
            data_files.extend(input_dir.rglob(ext))
        data_files = [f for f in data_files if "morfessor" not in str(f)]
        data_files.sort()
        base_dir = input_dir

    if not data_files:
        logger.error("No data files found in %s", input_dir)
        sys.exit(1)

    # ---- Build cache + load model ONCE in parent process ----
    logger.info("Loading model and building segmentation cache (once)...")
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    cache = _build_segmentation_cache(model, freq_path, separator)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    # ---- Process each file ----
    for fpath in data_files:
        try:
            rel = fpath.relative_to(base_dir)
        except ValueError:
            rel = Path(fpath.name)

        out_file = seg_dir / rel.with_suffix(".seg.txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already segmented
        if out_file.exists() and out_file.stat().st_size > 1024:
            logger.info("SKIPPING %s — already segmented", rel)
            continue

        logger.info("Segmenting %s -> %s (%d workers)", rel, out_file.name, num_workers)
        start = time.time()
        doc_count = 0

        if num_workers <= 1:
            # ---- Sequential: simple loop with progress bar ----
            with open(out_file, "w", encoding="utf-8") as fout:
                for text in tqdm(
                    _iter_text_from_file(fpath),
                    desc=fpath.name,
                    unit=" docs",
                ):
                    segmented_text = _segment_text_cached(
                        cache, model, text, separator
                    )
                    fout.write(segmented_text + "\n")
                    doc_count += 1
        else:
            # ---- Parallel: stream docs through imap_unordered for responsiveness ----
            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(cache, model, separator),
            ) as pool, open(out_file, "w", encoding="utf-8") as fout:

                pbar = tqdm(desc=fpath.name, unit=" docs")

                # imap_unordered streams results as they complete — no blocking on slow docs
                for seg_text in pool.imap_unordered(
                    _segment_single,
                    _iter_text_from_file(fpath),
                    chunksize=200,
                ):
                    fout.write(seg_text + "\n")
                    doc_count += 1
                    pbar.update(1)

                    # Flush periodically so file size visibly grows
                    if doc_count % 50000 == 0:
                        fout.flush()

                pbar.close()

        elapsed = time.time() - start
        size_mb = out_file.stat().st_size / (1024 ** 2)
        logger.info(
            "  Done: %d docs, %.1f MB, %.1f min",
            doc_count, size_mb, elapsed / 60,
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(output_dir: Path):
    """Print summary of output files."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("MORFESSOR OUTPUT SUMMARY")
    logger.info("=" * 70)

    total_size = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_size += size
            if size > 1024 ** 3:
                size_str = f"{size / (1024 ** 3):.2f} GB"
            elif size > 1024 ** 2:
                size_str = f"{size / (1024 ** 2):.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"
            logger.info("  %-55s  %s", str(path.relative_to(output_dir)), size_str)

    logger.info("-" * 70)
    total_str = f"{total_size / (1024 ** 3):.2f} GB" if total_size > 1024 ** 3 else f"{total_size / (1024 ** 2):.1f} MB"
    logger.info("  Total: %s", total_str)
    logger.info("  Location: %s", output_dir.resolve())
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Telugu morphological segmentation with Morfessor Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./data                           # Full pipeline
  %(prog)s --input ./data --train-only              # Train + inspect only
  %(prog)s --input ./data --segment-only \\
           --model ./data/morfessor/morfessor_telugu.bin   # Segment with existing model
  %(prog)s --input ./data --sample-size 10000000    # Train on 10M word tokens
  %(prog)s --input ./data --corpus-weight 2.0       # Less segmentation
  %(prog)s --input ./data --corpus-weight 0.5       # More segmentation
  %(prog)s --input ./data --separator " "            # Space-separated morphemes
  %(prog)s --input ./data --vocab-stats             # Show morpheme vocab stats
  %(prog)s --segment-only --vocab-stats             # Stats on existing model only
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="./data",
        help="Input data directory (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <input>/morfessor)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5_000_000,
        help="Number of word tokens to sample for training (default: 5M)",
    )
    parser.add_argument(
        "--corpus-weight",
        type=float,
        default=1.0,
        help="Morfessor corpus weight: higher = less segmentation, lower = more (default: 1.0)",
    )
    parser.add_argument(
        "--dampening",
        choices=["log", "ones", "none"],
        default="log",
        help="Frequency dampening: 'log' (recommended), 'ones' (ignore freq), 'none' (raw freq) (default: log)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="@@",
        help="Morpheme boundary marker for segmented output (default: @@)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only build word frequencies, train model, and inspect — skip full corpus segmentation",
    )
    parser.add_argument(
        "--segment-only",
        action="store_true",
        help="Only segment corpus using an existing model (requires --model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a pre-trained Morfessor model (.bin) for --segment-only mode",
    )
    parser.add_argument(
        "--freq-threshold",
        type=int,
        default=2,
        help="Minimum word frequency to include in training (default: 2)",
    )
    parser.add_argument(
        "--vocab-stats",
        action="store_true",
        help="Compute and display morpheme vocabulary statistics (can be used standalone with --model)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers for corpus segmentation (default: auto = cpu_count - 1)",
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input path does not exist: %s", input_dir)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "morfessor"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    if args.vocab_stats and args.segment_only:
        # Standalone vocab stats mode
        model_path = Path(args.model) if args.model else output_dir / "morfessor_telugu.bin"
        freq_path = output_dir / "word_frequencies.txt"
        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            sys.exit(1)
        if not freq_path.exists():
            logger.error("Word frequency file not found: %s", freq_path)
            sys.exit(1)
        compute_vocab_stats(model_path, freq_path, output_dir)

    elif args.segment_only:
        # Segment-only mode
        model_path = Path(args.model) if args.model else output_dir / "morfessor_telugu.bin"
        if not model_path.exists():
            logger.error("Model file not found: %s", model_path)
            logger.error("Train a model first or provide --model path.")
            sys.exit(1)

        logger.info("=" * 70)
        logger.info("Morfessor Segmentation (segment-only mode)")
        logger.info("  Model:     %s", model_path)
        logger.info("  Input:     %s", input_dir)
        logger.info("  Output:    %s", output_dir)
        logger.info("  Separator: '%s'", args.separator)
        logger.info("=" * 70)

        segment_corpus(input_dir, model_path, output_dir, args.separator, args.workers)

    else:
        # Full pipeline or train-only
        logger.info("=" * 70)
        logger.info("Morfessor Telugu Pipeline")
        logger.info("=" * 70)
        logger.info("  Input dir:     %s", input_dir)
        logger.info("  Output dir:    %s", output_dir)
        logger.info("  Sample size:   %d tokens", args.sample_size)
        logger.info("  Corpus weight: %.2f", args.corpus_weight)
        logger.info("  Dampening:     %s", args.dampening)
        logger.info("  Separator:     '%s'", args.separator)
        logger.info("  Mode:          %s", "train-only" if args.train_only else "full pipeline")
        logger.info("=" * 70)

        # Step 1: Build word frequencies
        freq_path = build_word_frequencies(input_dir, args.sample_size, output_dir)

        # Step 2: Train model
        model_path = train_morfessor(
            freq_path, output_dir, args.corpus_weight, args.dampening,
        )

        # Step 3: Inspect
        inspect_segmentation(model_path, output_dir)

        # Step 3b: Vocab stats (if requested or always during train-only)
        if args.vocab_stats or args.train_only:
            compute_vocab_stats(model_path, freq_path, output_dir)

        # Step 4: Segment full corpus (unless train-only)
        if not args.train_only:
            segment_corpus(input_dir, model_path, output_dir, args.separator, args.workers)

    elapsed_total = time.time() - start_total
    print_summary(output_dir)
    logger.info("Completed in %.1f minutes!", elapsed_total / 60)


if __name__ == "__main__":
    main()
