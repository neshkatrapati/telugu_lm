#!/usr/bin/env python3
"""
Telugu Morphological Segmentation with Morfessor Baseline
==========================================================
Trains a Morfessor Baseline model on Telugu text and applies segmentation.

v4: Reversed @@ prefix scheme
  - Clean split at Telugu/non-Telugu script boundaries BEFORE segmentation
  - Morfessor segments only pure Telugu words
  - First morpheme of each word is bare (root/stem)
  - Subsequent morphemes get @@ prefix if they appear as suffix often enough
  - Non-Telugu tokens kept as-is, separated by whitespace

  Example:
    "విద్యార్థులకు went 2024లో"
    → split scripts: "విద్యార్థులకు went 2024 లో"
    → segment: "విద్యార్థు @@ల @@కు went 2024 లో"
    → decode: concatenate tokens, @@ prefix = join to previous, else new word
    → "విద్యార్థులకు went 2024 లో"

Pipeline:
  1. Extract Telugu words + frequencies (with script-boundary cleanup)
  2. Train Morfessor Baseline model
  3. Inspect segmentation quality
  4. Pass 1: Segment corpus, collect suffix position frequencies
  5. Pass 2: Re-segment with @@ prefix on qualifying suffixes

Requirements:
    pip install morfessor tqdm

Usage:
    # Full pipeline: train + segment
    python morfessor_segment.py --input ./data

    # Train only (inspect before segmenting everything)
    python morfessor_segment.py --input ./data --train-only

    # Segment only (using a previously trained model)
    python morfessor_segment.py --input ./data --segment-only --model ./data/morfessor/morfessor_telugu.bin

    # Custom corpus weight (higher = less segmentation, lower = more)
    python morfessor_segment.py --input ./data --corpus-weight 0.5

    # Set min suffix coverage (default: 99.0%)
    python morfessor_segment.py --input ./data --suffix-coverage 99.5
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
TELUGU_CHAR_RE = re.compile(r"[\u0C00-\u0C7F]")
TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")

# Script boundary split: split at transitions between Telugu and non-Telugu
# This regex matches the zero-width boundary between the two script classes
SCRIPT_BOUNDARY_RE = re.compile(
    r"(?<=[^\u0C00-\u0C7F])(?=[\u0C00-\u0C7F])"
    r"|"
    r"(?<=[\u0C00-\u0C7F])(?=[^\u0C00-\u0C7F])"
)


def is_telugu(text: str) -> bool:
    """Check if text contains Telugu characters."""
    return bool(TELUGU_CHAR_RE.search(text))


def extract_telugu_words(text: str) -> list[str]:
    """Extract Telugu words from a line of text."""
    return TELUGU_WORD_RE.findall(text)


def split_script_boundaries(token: str) -> list[str]:
    """Split a token at Telugu/non-Telugu script boundaries.

    "తెలుసా..?" → ["తెలుసా", "..?"]
    "2024లో" → ["2024", "లో"]
    "IPLలో" → ["IPL", "లో"]
    "hello" → ["hello"]
    "విద్యార్థులకు" → ["విద్యార్థులకు"]
    """
    parts = SCRIPT_BOUNDARY_RE.split(token)
    return [p for p in parts if p]


def clean_text_script_boundaries(text: str) -> str:
    """Split all tokens at script boundaries, producing clean tokens.

    "తెలుసా..? 2024లో" → "తెలుసా ..? 2024 లో"
    """
    result = []
    for token in text.split():
        result.extend(split_script_boundaries(token))
    return " ".join(result)


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

    v4: Applies script-boundary cleanup before extracting Telugu words,
    so "తెలుసా..?" is split into "తెలుసా" + "..?" and only "తెలుసా" is counted.
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
                # v4: Clean script boundaries before extracting Telugu words
                cleaned = clean_text_script_boundaries(text)
                words = extract_telugu_words(cleaned)
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
    freq_threshold: int = 2,
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
    logger.info("  Freq threshold: %d", freq_threshold)

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
    model.load_data(word_counts, freqthreshold=freq_threshold, count_modifier=count_modifier)

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
# Step 4: Segment full corpus
# ---------------------------------------------------------------------------

def _build_segmentation_cache(model, freq_path: Path) -> dict[str, list[str]]:
    """
    Pre-segment all known words into a lookup dict.
    Returns dict mapping word → list of bare morphemes.
    """
    from tqdm import tqdm

    cache = {}

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
            cache[word] = list(segments)

    logger.info("Cached segmentations for %d word types", len(cache))
    return cache


MAX_TOKEN_LEN = 80


def _segment_telugu_word(cache: dict, model, word: str) -> list[str]:
    """Segment a pure-Telugu word into bare morphemes using cache.

    Returns list of bare morphemes, e.g. ["విద్యార్థు", "ల", "కు"]
    """
    cached = cache.get(word)
    if cached is not None:
        return list(cached)

    if len(word) > MAX_TOKEN_LEN:
        cache[word] = [word]
        return [word]

    segments = list(model.viterbi_segment(word)[0])
    cache[word] = segments
    return segments


def _segment_text_bare(cache: dict, model, text: str) -> list[list[str]]:
    """Segment text into a list of words, each being a list of bare morphemes.

    v4: Applies script-boundary cleanup. Each token is split at Telugu/non-Telugu
    boundaries first. Pure Telugu parts get Morfessor segmentation.
    Non-Telugu parts are kept as single-morpheme words.

    Returns list of words, where each word is a list of morpheme strings.
    Example: "విద్యార్థులకు went" → [["విద్యార్థు", "ల", "కు"], ["went"]]
    """
    words = []
    for token in text.split():
        # Split at script boundaries
        parts = split_script_boundaries(token)
        for part in parts:
            if TELUGU_WORD_RE.fullmatch(part):
                morphemes = _segment_telugu_word(cache, model, part)
                words.append(morphemes)
            else:
                words.append([part])
    return words


def _format_word_v4(morphemes: list[str], suffix_set: set[str] | None) -> list[str]:
    """Format a single word's morphemes in v4 reversed @@ format.

    First morpheme is bare. Subsequent morphemes get @@ prefix if they
    are in suffix_set (or if suffix_set is None → all get @@).

    Args:
        morphemes: List of bare morphemes for one word.
        suffix_set: Set of morphemes that qualify for @@ prefix form.
                    None = all non-initial morphemes get @@.

    Returns list of formatted tokens.
    """
    if len(morphemes) <= 1:
        return morphemes

    result = [morphemes[0]]  # first morpheme always bare
    for m in morphemes[1:]:
        if suffix_set is None or m in suffix_set:
            result.append("@@" + m)
        else:
            # No @@ form — force word break (treat as standalone)
            result.append(m)
    return result


def _segment_text_v4(cache: dict, model, text: str, suffix_set: set[str] | None) -> str:
    """Segment text in v4 reversed @@ format.

    Args:
        cache: Segmentation cache.
        model: Morfessor model.
        text: Raw input text.
        suffix_set: Set of morphemes qualifying for @@ prefix.
                    None = all non-initial morphemes get @@.

    Returns space-separated token string in v4 format.
    """
    words = _segment_text_bare(cache, model, text)
    result = []
    for morphemes in words:
        result.extend(_format_word_v4(morphemes, suffix_set))
    return " ".join(result)


# ---------------------------------------------------------------------------
# Pass 1: Collect suffix position frequencies
# ---------------------------------------------------------------------------

def _count_positions_single(text):
    """Worker function: segment one text and return position counts.

    Returns (solo, initial, cont, final) as dicts.
    """
    try:
        if len(text) > MAX_DOC_CHARS:
            return {}, {}, {}, {}
        words = _segment_text_bare(_shared_cache, _shared_model, text)
        solo = {}
        initial = {}
        cont = {}
        final = {}
        for morphemes in words:
            if len(morphemes) == 1:
                if TELUGU_CHAR_RE.search(morphemes[0]):
                    solo[morphemes[0]] = solo.get(morphemes[0], 0) + 1
            else:
                for idx, m in enumerate(morphemes):
                    if not TELUGU_CHAR_RE.search(m):
                        continue
                    if idx == 0:
                        initial[m] = initial.get(m, 0) + 1
                    elif idx == len(morphemes) - 1:
                        final[m] = final.get(m, 0) + 1
                    else:
                        cont[m] = cont.get(m, 0) + 1
        return solo, initial, cont, final
    except Exception:
        return {}, {}, {}, {}


def collect_suffix_frequencies(
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
    num_workers: int = 0,
    num_docs: int = 0,
) -> tuple[Counter, Counter, Counter, Counter]:
    """Segment corpus and count morpheme positions: solo, initial, cont, final.

    Parallelized: uses multiprocessing Pool for throughput.

    Returns (solo_freq, initial_freq, cont_freq, final_freq) Counters.
    Each maps morpheme_base → count in that position.
    """
    import morfessor
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    logger.info("Pass 1: Collecting suffix position frequencies...")

    # Load model and cache
    freq_path = output_dir / "word_frequencies.txt"
    if not freq_path.exists():
        freq_path = model_path.parent / "word_frequencies.txt"

    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    cache = _build_segmentation_cache(model, freq_path)

    # Find data files
    if input_dir.is_file():
        data_files = [input_dir]
    else:
        data_files = []
        for ext in ("*.parquet", "*.jsonl", "*.txt"):
            data_files.extend(input_dir.rglob(ext))
        data_files = [f for f in data_files if "morfessor" not in str(f)]
        data_files.sort()

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    solo_freq = Counter()
    initial_freq = Counter()
    cont_freq = Counter()
    final_freq = Counter()
    doc_count = 0

    for fpath in data_files:
        if num_workers <= 1:
            # Sequential
            for text in tqdm(
                _iter_text_from_file(fpath),
                desc=f"Pass 1: {fpath.name}",
                unit=" docs",
                total=num_docs if num_docs > 0 else None,
            ):
                words = _segment_text_bare(cache, model, text)
                for morphemes in words:
                    if len(morphemes) == 1:
                        if is_telugu(morphemes[0]):
                            solo_freq[morphemes[0]] += 1
                    else:
                        for idx, m in enumerate(morphemes):
                            if not is_telugu(m):
                                continue
                            if idx == 0:
                                initial_freq[m] += 1
                            elif idx == len(morphemes) - 1:
                                final_freq[m] += 1
                            else:
                                cont_freq[m] += 1

                doc_count += 1
                if num_docs > 0 and doc_count >= num_docs:
                    break
        else:
            # Parallel
            with Pool(
                processes=num_workers,
                initializer=_init_worker_pass1,
                initargs=(cache, model),
            ) as pool:
                pbar = tqdm(desc=f"Pass 1: {fpath.name}", unit=" docs",
                            total=num_docs if num_docs > 0 else None)

                for solo, initial, cont, final in pool.imap_unordered(
                    _count_positions_single,
                    _iter_text_from_file(fpath),
                    chunksize=1,
                ):
                    for m, c in solo.items():
                        solo_freq[m] += c
                    for m, c in initial.items():
                        initial_freq[m] += c
                    for m, c in cont.items():
                        cont_freq[m] += c
                    for m, c in final.items():
                        final_freq[m] += c

                    doc_count += 1
                    pbar.update(1)

                    if num_docs > 0 and doc_count >= num_docs:
                        pool.terminate()
                        break

                pbar.close()

        if num_docs > 0 and doc_count >= num_docs:
            break

    logger.info("Pass 1 complete: %d documents processed", doc_count)

    # Stats
    all_morphs = set(solo_freq) | set(initial_freq) | set(cont_freq) | set(final_freq)
    logger.info("  Unique Telugu morphemes: %d", len(all_morphs))
    logger.info("  Solo: %d types, Initial: %d types, Cont: %d types, Final: %d types",
                len(solo_freq), len(initial_freq), len(cont_freq), len(final_freq))

    return solo_freq, initial_freq, cont_freq, final_freq


def compute_suffix_set(
    solo_freq: Counter,
    initial_freq: Counter,
    cont_freq: Counter,
    final_freq: Counter,
    target_coverage: float = 99.0,
) -> tuple[set[str], int]:
    """Determine which morphemes get @@ prefix form based on target suffix coverage.

    A morpheme needs @@ prefix form if it appears as cont or final in multi-morpheme words.
    We find the minimum suffix frequency threshold that achieves target_coverage% of all
    suffix occurrences.

    Args:
        solo_freq, initial_freq, cont_freq, final_freq: Position counters.
        target_coverage: Target percentage of suffix occurrences to cover (default: 99.0).

    Returns:
        (suffix_set, threshold): Set of morphemes qualifying for @@ prefix, and the
        min_suffix threshold used.
    """
    # Count suffix occurrences per morpheme
    suffix_occ = Counter()
    all_morphs = set(cont_freq) | set(final_freq)
    for m in all_morphs:
        suffix_occ[m] = cont_freq[m] + final_freq[m]

    total_suffix = sum(suffix_occ.values())
    if total_suffix == 0:
        logger.warning("No suffix occurrences found!")
        return set(), 0

    # Sort morphemes by their suffix frequency descending
    sorted_morphs = sorted(suffix_occ.items(), key=lambda x: x[1], reverse=True)

    # Find threshold that covers target_coverage%
    cumulative = 0
    threshold = 0
    for m, count in sorted_morphs:
        cumulative += count
        coverage = 100 * cumulative / total_suffix
        if coverage >= target_coverage:
            threshold = count
            break

    # Build suffix set: all morphemes with suffix_occ >= threshold
    suffix_set = {m for m, c in suffix_occ.items() if c >= threshold}

    # Compute actual stats
    covered = sum(c for m, c in suffix_occ.items() if m in suffix_set)
    actual_coverage = 100 * covered / total_suffix

    all_tel = set(solo_freq) | set(initial_freq) | set(cont_freq) | set(final_freq)
    total_freq = Counter()
    for m in all_tel:
        total_freq[m] = solo_freq[m] + initial_freq[m] + cont_freq[m] + final_freq[m]
    total_all = sum(total_freq.values())
    missed = total_suffix - covered

    logger.info("Suffix set computation:")
    logger.info("  Target coverage: %.1f%%", target_coverage)
    logger.info("  Min suffix threshold: %d", threshold)
    logger.info("  Morphemes with @@ form: %d", len(suffix_set))
    logger.info("  Total unique Telugu morphemes: %d", len(all_tel))
    logger.info("  Total vocab entries (bare + @@): %d", len(all_tel) + len(suffix_set))
    logger.info("  Suffix coverage: %.2f%% (%d / %d)", actual_coverage, covered, total_suffix)
    logger.info("  Missed suffix occurrences: %d (%.3f%% of all tokens)",
                missed, 100 * missed / total_all if total_all else 0)

    # Save suffix set
    return suffix_set, threshold


# ---------------------------------------------------------------------------
# Pass 2: Segment corpus with final v4 format
# ---------------------------------------------------------------------------

# Module-level globals for shared state across forked workers
_shared_cache = None
_shared_model = None
_shared_suffix_set = None


def _init_worker_pass1(cache, model):
    """Initializer for Pass 1 pool workers — sets cache and model only."""
    global _shared_cache, _shared_model
    _shared_cache = cache
    _shared_model = model


def _init_worker(cache, model, suffix_set):
    """Initializer for Pass 2 pool workers — sets shared globals from parent."""
    global _shared_cache, _shared_model, _shared_suffix_set
    _shared_cache = cache
    _shared_model = model
    _shared_suffix_set = suffix_set


MAX_DOC_CHARS = 500_000


def _segment_single(text):
    """Worker function: segment a single text in v4 format."""
    try:
        if len(text) > MAX_DOC_CHARS:
            return text
        return _segment_text_v4(_shared_cache, _shared_model, text, _shared_suffix_set)
    except Exception:
        return text


def segment_corpus(
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
    suffix_set: set[str],
    num_workers: int = 0,
    num_docs: int = 0,
):
    """
    Pass 2: Segment full corpus in v4 reversed @@ format.
    """
    import morfessor
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    seg_dir = output_dir
    seg_dir.mkdir(parents=True, exist_ok=True)

    freq_path = output_dir / "word_frequencies.txt"
    if not freq_path.exists():
        freq_path = model_path.parent / "word_frequencies.txt"

    # Find data files
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

    # Load model and cache
    logger.info("Pass 2: Loading model and building segmentation cache...")
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    cache = _build_segmentation_cache(model, freq_path)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    logger.info("Suffix set: %d morphemes with @@ prefix form", len(suffix_set))

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
            with open(out_file, "w", encoding="utf-8") as fout:
                for text in tqdm(
                    _iter_text_from_file(fpath),
                    desc=fpath.name,
                    unit=" docs",
                    total=num_docs if num_docs > 0 else None,
                ):
                    segmented_text = _segment_text_v4(cache, model, text, suffix_set)
                    fout.write(segmented_text + "\n")
                    doc_count += 1
                    if num_docs > 0 and doc_count >= num_docs:
                        break
        else:
            with Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(cache, model, suffix_set),
            ) as pool, open(out_file, "w", encoding="utf-8") as fout:

                pbar = tqdm(desc=fpath.name, unit=" docs",
                            total=num_docs if num_docs > 0 else None)

                for seg_text in pool.imap_unordered(
                    _segment_single,
                    _iter_text_from_file(fpath),
                    chunksize=1,
                ):
                    fout.write(seg_text + "\n")
                    doc_count += 1
                    pbar.update(1)

                    if doc_count % 50000 == 0:
                        fout.flush()

                    if num_docs > 0 and doc_count >= num_docs:
                        pool.terminate()
                        break

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
        description="Telugu morphological segmentation with Morfessor Baseline (v4: reversed @@ prefix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./data                           # Full pipeline
  %(prog)s --input ./data --train-only              # Train + inspect only
  %(prog)s --input ./data --segment-only \\
           --model ./data/morfessor/morfessor_telugu.bin   # Segment with existing model
  %(prog)s --input ./data --corpus-weight 0.5       # More segmentation
  %(prog)s --input ./data --suffix-coverage 99.5    # Higher suffix coverage
        """,
    )

    parser.add_argument(
        "--input", type=str, default="./data",
        help="Input data directory (default: ./data)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: <input>/morfessor)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5_000_000,
        help="Number of word tokens to sample for training (default: 5M)",
    )
    parser.add_argument(
        "--corpus-weight", type=float, default=1.0,
        help="Morfessor corpus weight: higher = less segmentation, lower = more (default: 1.0)",
    )
    parser.add_argument(
        "--dampening", choices=["log", "ones", "none"], default="log",
        help="Frequency dampening (default: log)",
    )
    parser.add_argument(
        "--freq-threshold", type=int, default=2,
        help="Minimum word frequency for Morfessor training (default: 2)",
    )
    parser.add_argument(
        "--suffix-coverage", type=float, default=99.0,
        help="Target suffix coverage percentage for @@ prefix selection (default: 99.0)",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only build word frequencies, train model, and inspect",
    )
    parser.add_argument(
        "--segment-only", action="store_true",
        help="Only segment corpus using an existing model (requires --model)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a pre-trained Morfessor model (.bin)",
    )
    parser.add_argument(
        "--vocab-stats", action="store_true",
        help="Compute and display morpheme vocabulary statistics",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (default: auto = cpu_count - 1)",
    )
    parser.add_argument(
        "--num-docs", type=int, default=0,
        help="Limit to first N documents (0 = all, default: 0)",
    )

    args = parser.parse_args()

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
            sys.exit(1)

        logger.info("=" * 70)
        logger.info("Morfessor Segmentation v4 (reversed @@ prefix)")
        logger.info("  Model:           %s", model_path)
        logger.info("  Input:           %s", input_dir)
        logger.info("  Output:          %s", output_dir)
        logger.info("  Suffix coverage: %.1f%%", args.suffix_coverage)
        logger.info("=" * 70)

        # Check if suffix_set.json already exists (from a previous sample run)
        suffix_path = output_dir / "suffix_set.json"
        if suffix_path.exists():
            logger.info("Loading existing suffix set from %s", suffix_path)
            with open(suffix_path, "r", encoding="utf-8") as f:
                suffix_data = json.load(f)
            suffix_set = set(suffix_data["morphemes"])
            threshold = suffix_data["threshold"]
            logger.info("  Loaded %d morphemes (threshold=%d, coverage=%.1f%%)",
                        len(suffix_set), threshold, suffix_data["coverage"])
        else:
            # Pass 1: Collect suffix frequencies
            solo_freq, initial_freq, cont_freq, final_freq = collect_suffix_frequencies(
                input_dir, model_path, output_dir, args.workers, args.num_docs,
            )

            # Compute suffix set
            suffix_set, threshold = compute_suffix_set(
                solo_freq, initial_freq, cont_freq, final_freq,
                target_coverage=args.suffix_coverage,
            )

            # Save suffix set for reference
            with open(suffix_path, "w", encoding="utf-8") as f:
                json.dump({
                    "threshold": threshold,
                    "coverage": args.suffix_coverage,
                    "count": len(suffix_set),
                    "morphemes": sorted(suffix_set),
                }, f, ensure_ascii=False, indent=2)
            logger.info("Saved suffix set (%d morphemes) to %s", len(suffix_set), suffix_path)

        # Pass 2: Segment with v4 format
        segment_corpus(input_dir, model_path, output_dir, suffix_set, args.workers, args.num_docs)

    else:
        # Full pipeline or train-only
        logger.info("=" * 70)
        logger.info("Morfessor Telugu Pipeline v4")
        logger.info("=" * 70)
        logger.info("  Input dir:       %s", input_dir)
        logger.info("  Output dir:      %s", output_dir)
        logger.info("  Sample size:     %d tokens", args.sample_size)
        logger.info("  Corpus weight:   %.2f", args.corpus_weight)
        logger.info("  Dampening:       %s", args.dampening)
        logger.info("  Freq threshold:  %d", args.freq_threshold)
        logger.info("  Suffix coverage: %.1f%%", args.suffix_coverage)
        logger.info("  Mode:            %s", "train-only" if args.train_only else "full pipeline")
        logger.info("=" * 70)

        # Step 1: Build word frequencies (with script-boundary cleanup)
        freq_path = build_word_frequencies(input_dir, args.sample_size, output_dir)

        # Step 2: Train model
        model_path = train_morfessor(
            freq_path, output_dir, args.corpus_weight, args.dampening, args.freq_threshold,
        )

        # Step 3: Inspect
        inspect_segmentation(model_path, output_dir)

        # Step 3b: Vocab stats
        if args.vocab_stats or args.train_only:
            compute_vocab_stats(model_path, freq_path, output_dir)

        # Step 4+5: Segment full corpus (unless train-only)
        if not args.train_only:
            # Pass 1: Collect suffix frequencies
            solo_freq, initial_freq, cont_freq, final_freq = collect_suffix_frequencies(
                input_dir, model_path, output_dir, args.workers, args.num_docs,
            )

            # Compute suffix set
            suffix_set, threshold = compute_suffix_set(
                solo_freq, initial_freq, cont_freq, final_freq,
                target_coverage=args.suffix_coverage,
            )

            # Save suffix set
            suffix_path = output_dir / "suffix_set.json"
            with open(suffix_path, "w", encoding="utf-8") as f:
                json.dump({
                    "threshold": threshold,
                    "coverage": args.suffix_coverage,
                    "count": len(suffix_set),
                    "morphemes": sorted(suffix_set),
                }, f, ensure_ascii=False, indent=2)
            logger.info("Saved suffix set (%d morphemes) to %s", len(suffix_set), suffix_path)

            # Pass 2: Segment with v4 format
            segment_corpus(input_dir, model_path, output_dir, suffix_set, args.workers, args.num_docs)

    elapsed_total = time.time() - start_total
    print_summary(output_dir)
    logger.info("Completed in %.1f minutes!", elapsed_total / 60)


if __name__ == "__main__":
    main()
