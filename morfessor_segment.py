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

    # Find all data files
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
# Step 4: Segment full corpus
# ---------------------------------------------------------------------------
def segment_corpus(
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
    separator: str,
):
    """Apply Morfessor segmentation to all corpus files."""
    import morfessor
    from tqdm import tqdm

    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))

    seg_dir = output_dir / "segmented_corpus"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Find all data files (same logic as build_word_frequencies)
    data_files = []
    for ext in ("*.parquet", "*.jsonl", "*.txt"):
        data_files.extend(input_dir.rglob(ext))
    data_files = [f for f in data_files if "morfessor" not in str(f)]
    data_files.sort()

    if not data_files:
        logger.error("No data files found in %s", input_dir)
        sys.exit(1)

    logger.info("Segmenting %d files...", len(data_files))

    for fpath in data_files:
        # Build output filename preserving relative structure
        try:
            rel = fpath.relative_to(input_dir)
        except ValueError:
            rel = Path(fpath.name)

        out_file = seg_dir / rel.with_suffix(".seg.txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already segmented
        if out_file.exists() and out_file.stat().st_size > 1024:
            logger.info("SKIPPING %s — already segmented", rel)
            continue

        logger.info("Segmenting %s -> %s", rel, out_file.name)
        start = time.time()
        doc_count = 0

        with open(out_file, "w", encoding="utf-8") as fout:
            for text in tqdm(
                _iter_text_from_file(fpath),
                desc=fpath.name,
                unit=" docs",
            ):
                segmented_text = _segment_text(model, text, separator)
                fout.write(segmented_text + "\n")
                doc_count += 1

        elapsed = time.time() - start
        size_mb = out_file.stat().st_size / (1024 ** 2)
        logger.info(
            "  Done: %d docs, %.1f MB, %.1f min",
            doc_count, size_mb, elapsed / 60,
        )


def _segment_text(model, text: str, separator: str) -> str:
    """
    Segment Telugu words in text, leaving non-Telugu tokens untouched.

    Example (separator="@@"):
      "విద్యార్థులకు went to school" -> "విద్యార్థు@@ ల@@ కు went to school"
    """
    tokens = text.split()
    result = []

    for token in tokens:
        # Extract Telugu parts
        telugu_words = TELUGU_WORD_RE.findall(token)
        if not telugu_words:
            # No Telugu characters — keep as-is
            result.append(token)
            continue

        # If the token is purely Telugu, segment it
        if TELUGU_WORD_RE.fullmatch(token):
            segments = model.viterbi_segment(token)[0]
            if len(segments) > 1:
                # Mark continuation with separator (like BPE)
                segmented = []
                for i, seg in enumerate(segments):
                    if i < len(segments) - 1:
                        segmented.append(seg + separator)
                    else:
                        segmented.append(seg)
                result.extend(segmented)
            else:
                result.append(token)
        else:
            # Mixed token — segment Telugu parts, keep rest
            # Split on Telugu boundaries
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            for part in parts:
                if not part:
                    continue
                if TELUGU_WORD_RE.fullmatch(part):
                    segments = model.viterbi_segment(part)[0]
                    if len(segments) > 1:
                        for i, seg in enumerate(segments):
                            if i < len(segments) - 1:
                                result.append(seg + separator)
                            else:
                                result.append(seg)
                    else:
                        result.append(part)
                else:
                    result.append(part)

    return " ".join(result)


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

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "morfessor"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    if args.segment_only:
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

        segment_corpus(input_dir, model_path, output_dir, args.separator)

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

        # Step 4: Segment full corpus (unless train-only)
        if not args.train_only:
            segment_corpus(input_dir, model_path, output_dir, args.separator)

    elapsed_total = time.time() - start_total
    print_summary(output_dir)
    logger.info("Completed in %.1f minutes!", elapsed_total / 60)


if __name__ == "__main__":
    main()
