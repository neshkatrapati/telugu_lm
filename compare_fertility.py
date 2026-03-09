#!/usr/bin/env python3
"""
Compare tokenizer fertility: Morfessor v4 vs SentencePiece.

Fertility = tokens per word (lower is better — fewer tokens to represent same text).

Reads raw Telugu text (parquet or .txt), tokenizes with both models,
and reports:
  - Total tokens produced
  - Fertility (tokens / whitespace words)
  - UNK rate
  - Token length distribution
  - Per-sentence breakdown (optional)

Usage:
    python compare_fertility.py \
        --input data/sample.parquet \
        --morf-tokenizer ./tokenizer \
        --sp-model ./sp_tokenizer/sp_telugu.model \
        --num-docs 50000
"""

import re
import sys
import json
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

TELUGU_CHAR_RE = re.compile(r"[\u0C00-\u0C7F]")

def has_telugu(s: str) -> bool:
    return bool(TELUGU_CHAR_RE.search(s))


def iter_texts(input_path: Path, num_docs: int = 0):
    """Yield raw text strings from a file (parquet, jsonl, or txt)."""
    suffix = input_path.suffix.lower()
    count = 0

    if suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(input_path)
        for batch in pf.iter_batches(batch_size=5000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text and text.strip():
                    yield text.strip()
                    count += 1
                    if num_docs > 0 and count >= num_docs:
                        return
    elif suffix == ".jsonl":
        import json as _json
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = _json.loads(line)
                    text = row.get("text", "").strip()
                    if text:
                        yield text
                        count += 1
                        if num_docs > 0 and count >= num_docs:
                            return
                except Exception:
                    continue
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
                    count += 1
                    if num_docs > 0 and count >= num_docs:
                        return


def load_morf_tokenizer(tokenizer_dir: Path, morf_model: str = None, suffix_set_path: str = None):
    """Load Morfessor v4 tokenizer + model for segmentation."""
    sys.path.insert(0, str(tokenizer_dir.parent))
    from train_tokenizer import MorfessorTokenizer
    from morfessor_segment import (
        split_script_boundaries, TELUGU_WORD_RE,
        _segment_telugu_word, _format_word_v4,
    )
    import morfessor

    tokenizer = MorfessorTokenizer(tokenizer_dir)

    # Load Morfessor model for segmentation
    model = None
    suffix_set = None

    search_model = [Path("./data/morfessor/morfessor_telugu.bin"),
                    Path("./morfessor_telugu.bin")]
    if morf_model:
        search_model.insert(0, Path(morf_model))

    for mpath in search_model:
        if mpath.exists():
            io = morfessor.MorfessorIO()
            model = io.read_binary_model_file(str(mpath))
            logger.info("Loaded Morfessor model from %s", mpath)
            break

    # Load suffix set — search near the model too
    search_suffix = [Path("./data/morfessor/suffix_set.json"),
                     Path("./suffix_set.json")]
    if suffix_set_path:
        search_suffix.insert(0, Path(suffix_set_path))
    if model is not None:
        # Also look next to the model file
        for mpath in search_model:
            if mpath.exists():
                search_suffix.insert(0, mpath.parent / "suffix_set.json")
                break

    for spath in search_suffix:
        if spath.exists():
            with open(spath, "r", encoding="utf-8") as f:
                suffix_data = json.load(f)
            suffix_set = set(suffix_data["morphemes"])
            logger.info("Loaded suffix set from %s (%d morphemes)", spath, len(suffix_set))
            break

    if model is None:
        logger.error("Morfessor model not found! Searched: %s", [str(p) for p in search_model])
        sys.exit(1)
    if suffix_set is None:
        logger.error("suffix_set.json not found! Searched: %s", [str(p) for p in search_suffix])
        sys.exit(1)

    logger.info("Loaded Morfessor v4 tokenizer (vocab=%d, suffix_set=%d)",
                tokenizer.vocab_size, len(suffix_set))

    def encode_raw(text: str) -> list[int]:
        """Segment raw text with Morfessor, then encode with v4 tokenizer."""
        cache = {}
        seg_tokens = []
        for word in text.split():
            parts = split_script_boundaries(word)
            for part in parts:
                if TELUGU_WORD_RE.fullmatch(part):
                    morphemes = _segment_telugu_word(cache, model, part)
                    seg_tokens.extend(_format_word_v4(morphemes, suffix_set))
                else:
                    seg_tokens.append(part)
        segmented = " ".join(seg_tokens)
        return tokenizer.encode(segmented, add_bos=False, add_eos=False)

    return tokenizer, encode_raw


def load_sp_tokenizer(model_path: Path):
    """Load SentencePiece tokenizer."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    logger.info("Loaded SentencePiece tokenizer (vocab=%d)", sp.get_piece_size())

    def encode_raw(text: str) -> list[int]:
        return sp.encode(text, out_type=int)

    return sp, encode_raw


def compute_stats(texts, encode_fn, name: str, show_examples: int = 5):
    """Compute fertility stats for a tokenizer."""
    from tqdm import tqdm

    total_tokens = 0
    total_words = 0
    total_tel_tokens = 0
    total_tel_words = 0
    total_nontel_tokens = 0
    total_nontel_words = 0
    total_unk = 0
    doc_count = 0
    token_lengths = Counter()  # length in chars → count

    examples = []

    for text in tqdm(texts, desc=name, unit=" docs"):
        words = text.split()
        n_words = len(words)
        if n_words == 0:
            continue

        ids = encode_fn(text)
        n_tokens = len(ids)

        total_tokens += n_tokens
        total_words += n_words
        doc_count += 1

        # Count Telugu vs non-Telugu words
        tel_words = sum(1 for w in words if has_telugu(w))
        nontel_words = n_words - tel_words
        total_tel_words += tel_words
        total_nontel_words += nontel_words

        if len(examples) < show_examples:
            examples.append((text[:120], n_words, n_tokens, n_tokens / n_words))

    fertility = total_tokens / total_words if total_words > 0 else 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("  %s — Fertility Report", name)
    logger.info("=" * 70)
    logger.info("  Documents:        %d", doc_count)
    logger.info("  Total words:      %d", total_words)
    logger.info("  Total tokens:     %d", total_tokens)
    logger.info("  Fertility:        %.3f tokens/word", fertility)
    logger.info("  Telugu words:     %d (%.1f%%)", total_tel_words,
                100 * total_tel_words / total_words if total_words else 0)
    logger.info("  Non-Telugu words: %d (%.1f%%)", total_nontel_words,
                100 * total_nontel_words / total_words if total_words else 0)
    logger.info("")
    logger.info("  Sample documents:")
    for text, nw, nt, fert in examples:
        logger.info("    [%d words → %d tokens, fert=%.2f] %s...", nw, nt, fert, text[:80])
    logger.info("=" * 70)

    return {
        "name": name,
        "docs": doc_count,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "fertility": fertility,
        "telugu_words": total_tel_words,
        "nontelugu_words": total_nontel_words,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer fertility: Morf v4 vs SP")
    parser.add_argument("--input", type=str, required=True,
                        help="Raw text input (parquet, jsonl, or txt)")
    parser.add_argument("--morf-tokenizer", type=str, default="./tokenizer",
                        help="Path to Morfessor v4 tokenizer dir")
    parser.add_argument("--sp-model", type=str, default=None,
                        help="Path to SentencePiece .model file")
    parser.add_argument("--morf-model", type=str, default=None,
                        help="Path to morfessor_telugu.bin")
    parser.add_argument("--suffix-set", type=str, default=None,
                        help="Path to suffix_set.json")
    parser.add_argument("--num-docs", type=int, default=10000,
                        help="Number of documents to test (default: 10000)")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    # Collect texts once (to ensure both tokenizers see the same data)
    logger.info("Loading %d documents from %s...", args.num_docs, input_path)
    texts = list(iter_texts(input_path, args.num_docs))
    logger.info("Loaded %d documents", len(texts))

    results = []

    # --- Morfessor v4 ---
    morf_dir = Path(args.morf_tokenizer)
    if morf_dir.exists():
        tokenizer, morf_encode = load_morf_tokenizer(morf_dir, args.morf_model, args.suffix_set)
        stats = compute_stats(texts, morf_encode, "Morfessor v4")
        results.append(stats)
    else:
        logger.warning("Morfessor tokenizer not found at %s — skipping", morf_dir)

    # --- SentencePiece ---
    if args.sp_model:
        sp_path = Path(args.sp_model)
        if sp_path.exists():
            sp, sp_encode = load_sp_tokenizer(sp_path)
            stats = compute_stats(texts, sp_encode, "SentencePiece")
            results.append(stats)
        else:
            logger.warning("SP model not found at %s — skipping", sp_path)

    # --- Comparison ---
    if len(results) >= 2:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  COMPARISON")
        logger.info("=" * 70)
        m, s = results[0], results[1]
        logger.info("  %-25s  %-15s  %-15s", "", m["name"], s["name"])
        logger.info("  %-25s  %-15d  %-15d", "Total tokens", m["total_tokens"], s["total_tokens"])
        logger.info("  %-25s  %-15.3f  %-15.3f", "Fertility (tok/word)", m["fertility"], s["fertility"])
        ratio = m["total_tokens"] / s["total_tokens"] if s["total_tokens"] else 0
        logger.info("")
        if ratio > 1:
            logger.info("  Morfessor produces %.1f%% MORE tokens than SP", (ratio - 1) * 100)
        else:
            logger.info("  Morfessor produces %.1f%% FEWER tokens than SP", (1 - ratio) * 100)
        logger.info("  Token ratio (Morf/SP): %.3f", ratio)
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
