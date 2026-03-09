#!/usr/bin/env python3
"""
Compare tokenizer quality via KenLM n-gram perplexity.

1. Load raw text, split 90/10 train/test
2. Tokenize with Morfessor v4 and SentencePiece
3. Write token IDs as space-separated text (KenLM format)
4. Train 5-gram KenLM model with lmplz
5. Score test set, compute BPC

Requirements:
    pip install kenlm
    # KenLM binaries (lmplz, build_binary) must be on PATH
    # Install: https://github.com/kpu/kenlm

Usage:
    python compare_kenlm_bpc.py \
        --input data/sample.parquet \
        --morf-tokenizer ./tokenizer \
        --morf-model data/morfessor/morfessor_telugu.bin \
        --suffix-set data/morfessor/suffix_set.json \
        --sp-model ./sp_tokenizer/sp_telugu.model \
        --num-docs 1000000 \
        --order 5
"""

import math
import os
import sys
import json
import subprocess
import argparse
import logging
import random
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------
def iter_texts(input_path: Path, num_docs: int = 0):
    """Yield raw text strings from a file."""
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


# ---------------------------------------------------------------------------
# Tokenizer loaders
# ---------------------------------------------------------------------------
def load_morf_encoder(tokenizer_dir: Path, morf_model_path: str = None,
                      suffix_set_path: str = None):
    sys.path.insert(0, str(tokenizer_dir.parent))
    from train_tokenizer import MorfessorTokenizer
    from morfessor_segment import (
        split_script_boundaries, TELUGU_WORD_RE,
        _segment_telugu_word, _format_word_v4,
    )
    import morfessor

    tokenizer = MorfessorTokenizer(tokenizer_dir)

    model = None
    search = [Path("./data/morfessor/morfessor_telugu.bin")]
    if morf_model_path:
        search.insert(0, Path(morf_model_path))
    for p in search:
        if p.exists():
            model = morfessor.MorfessorIO().read_binary_model_file(str(p))
            logger.info("Loaded Morfessor model: %s", p)
            break

    suffix_set = None
    search_s = [Path("./data/morfessor/suffix_set.json")]
    if suffix_set_path:
        search_s.insert(0, Path(suffix_set_path))
    if model:
        for p in search:
            if p.exists():
                search_s.insert(0, p.parent / "suffix_set.json")
                break
    for p in search_s:
        if p.exists():
            with open(p) as f:
                suffix_set = set(json.load(f)["morphemes"])
            logger.info("Loaded suffix set: %s (%d)", p, len(suffix_set))
            break

    if not model:
        logger.error("Morfessor model not found"); sys.exit(1)
    if suffix_set is None:
        logger.error("suffix_set.json not found"); sys.exit(1)

    logger.info("Morfessor v4 tokenizer: vocab=%d", tokenizer.vocab_size)

    # Use a shared cache across all calls for speed
    shared_cache = {}

    def encode(text: str) -> list[int]:
        seg_tokens = []
        for word in text.split():
            parts = split_script_boundaries(word)
            for part in parts:
                if TELUGU_WORD_RE.fullmatch(part):
                    morphemes = _segment_telugu_word(shared_cache, model, part)
                    seg_tokens.extend(_format_word_v4(morphemes, suffix_set))
                else:
                    seg_tokens.append(part)
        segmented = " ".join(seg_tokens)
        return tokenizer.encode(segmented, add_bos=False, add_eos=False)

    return encode, tokenizer.vocab_size


def load_sp_encoder(model_path: Path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    logger.info("SentencePiece tokenizer: vocab=%d", sp.get_piece_size())

    def encode(text: str) -> list[int]:
        return sp.encode(text, out_type=int)

    return encode, sp.get_piece_size()


# ---------------------------------------------------------------------------
# Tokenize and write KenLM format
# ---------------------------------------------------------------------------
def tokenize_and_write(texts: list[str], encode_fn, output_path: Path,
                       name: str) -> tuple[int, int]:
    """Tokenize texts and write as space-separated token ID strings.

    Returns (total_tokens, total_chars).
    """
    from tqdm import tqdm

    total_tokens = 0
    total_chars = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for text in tqdm(texts, desc=f"Tokenizing {name}", unit=" docs"):
            ids = encode_fn(text)
            if not ids:
                continue
            # Write token IDs as space-separated string (one doc per line)
            f.write(" ".join(str(i) for i in ids) + "\n")
            total_tokens += len(ids)
            total_chars += len(text)

    logger.info("  %s: %d docs, %d tokens, %d chars, %.2f chars/tok",
                name, len(texts), total_tokens, total_chars,
                total_chars / total_tokens if total_tokens else 0)
    return total_tokens, total_chars


# ---------------------------------------------------------------------------
# KenLM training and scoring
# ---------------------------------------------------------------------------
def train_kenlm(train_path: Path, model_path: Path, order: int = 5):
    """Train a KenLM model using lmplz."""
    arpa_path = model_path.with_suffix(".arpa")

    logger.info("Training KenLM %d-gram on %s ...", order, train_path.name)

    # lmplz: train ARPA model
    cmd = f"lmplz -o {order} --discount_fallback < '{train_path}' > '{arpa_path}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("lmplz failed:\n%s", result.stderr)
        sys.exit(1)

    arpa_size = arpa_path.stat().st_size / (1024 ** 2)
    logger.info("  ARPA model: %.1f MB", arpa_size)

    # build_binary: convert to binary for fast loading
    cmd2 = f"build_binary '{arpa_path}' '{model_path}'"
    result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    if result2.returncode != 0:
        logger.warning("build_binary failed (will use ARPA directly):\n%s", result2.stderr)
        # Fall back to ARPA
        return arpa_path

    bin_size = model_path.stat().st_size / (1024 ** 2)
    logger.info("  Binary model: %.1f MB", bin_size)

    # Clean up ARPA
    arpa_path.unlink(missing_ok=True)
    return model_path


def score_kenlm(model_path: Path, test_path: Path, test_chars: int) -> dict:
    """Score test set with KenLM and compute BPC."""
    import kenlm

    logger.info("Scoring with KenLM: %s", model_path.name)

    model = kenlm.Model(str(model_path))

    total_log10_prob = 0.0
    total_tokens = 0
    n_oov = 0
    n_lines = 0

    from tqdm import tqdm

    with open(test_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scoring", unit=" docs"):
        line = line.strip()
        if not line:
            continue
        # KenLM scores a sentence and returns log10 probability
        score = model.score(line, bos=True, eos=True)
        total_log10_prob += score

        # Count tokens and OOVs
        for word, (prob, length, oov) in zip(line.split(), model.full_scores(line, bos=True, eos=True)):
            total_tokens += 1
            if oov:
                n_oov += 1
        # +1 for EOS token that KenLM adds
        total_tokens += 1
        n_lines += 1

    # Convert log10 to log2
    total_log2_prob = total_log10_prob * math.log2(10)

    # BPC = -total_log2_prob / total_characters
    bpc = -total_log2_prob / test_chars if test_chars > 0 else 0

    # Perplexity (per token)
    avg_log10 = total_log10_prob / total_tokens if total_tokens else 0
    ppl = 10 ** (-avg_log10)

    logger.info("  Lines: %d, Tokens: %d, OOV: %d (%.2f%%)",
                n_lines, total_tokens, n_oov,
                100 * n_oov / total_tokens if total_tokens else 0)
    logger.info("  Perplexity: %.2f", ppl)
    logger.info("  BPC: %.4f", bpc)

    return {
        "ppl": ppl,
        "bpc": bpc,
        "total_tokens": total_tokens,
        "total_chars": test_chars,
        "oov": n_oov,
        "oov_rate": n_oov / total_tokens if total_tokens else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare tokenizers via KenLM n-gram BPC")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--morf-tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--morf-model", type=str, default=None)
    parser.add_argument("--suffix-set", type=str, default=None)
    parser.add_argument("--sp-model", type=str, default=None)
    parser.add_argument("--num-docs", type=int, default=1_000_000)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--order", type=int, default=5,
                        help="N-gram order (default: 5)")
    parser.add_argument("--work-dir", type=str, default="./kenlm_compare",
                        help="Working directory for temp files")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check KenLM is available
    r = subprocess.run("which lmplz", shell=True, capture_output=True)
    if r.returncode != 0:
        logger.error("lmplz not found on PATH. Install KenLM:")
        logger.error("  sudo apt install libboost-all-dev cmake")
        logger.error("  git clone https://github.com/kpu/kenlm && cd kenlm")
        logger.error("  mkdir build && cd build && cmake .. && make -j4")
        logger.error("  export PATH=$PATH:$(pwd)/bin")
        sys.exit(1)

    try:
        import kenlm  # noqa: F401
    except ImportError:
        logger.error("kenlm Python module not found. Install: pip install kenlm")
        sys.exit(1)

    # Load texts
    logger.info("Loading %d documents from %s...", args.num_docs, input_path)
    texts = list(iter_texts(input_path, args.num_docs))
    logger.info("Loaded %d documents", len(texts))

    # Split train/test
    random.seed(args.seed)
    random.shuffle(texts)
    split = int(len(texts) * (1 - args.test_frac))
    train_texts = texts[:split]
    test_texts = texts[split:]
    logger.info("Train: %d docs, Test: %d docs", len(train_texts), len(test_texts))

    # Free memory
    del texts

    # --- Build encoders ---
    encoders = []
    morf_dir = Path(args.morf_tokenizer)
    if morf_dir.exists():
        enc, vs = load_morf_encoder(morf_dir, args.morf_model, args.suffix_set)
        encoders.append(("morf", "Morfessor v4", enc, vs))

    if args.sp_model:
        sp_path = Path(args.sp_model)
        if sp_path.exists():
            enc, vs = load_sp_encoder(sp_path)
            encoders.append(("sp", "SentencePiece", enc, vs))

    results = []

    for short_name, name, encode_fn, vocab_size in encoders:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  %s (vocab=%d)", name, vocab_size)
        logger.info("=" * 70)

        train_file = work_dir / f"{short_name}_train.txt"
        test_file = work_dir / f"{short_name}_test.txt"
        model_file = work_dir / f"{short_name}_{args.order}gram.bin"

        # Tokenize train
        train_tokens, train_chars = tokenize_and_write(
            train_texts, encode_fn, train_file, f"{name} train")

        # Tokenize test
        test_tokens, test_chars = tokenize_and_write(
            test_texts, encode_fn, test_file, f"{name} test")

        # Train KenLM
        actual_model = train_kenlm(train_file, model_file, order=args.order)

        # Score
        scores = score_kenlm(actual_model, test_file, test_chars)
        scores["name"] = name
        scores["vocab_size"] = vocab_size
        scores["train_tokens"] = train_tokens
        scores["test_tokens_raw"] = test_tokens
        scores["chars_per_token"] = test_chars / test_tokens if test_tokens else 0
        results.append(scores)

    # --- Comparison ---
    if len(results) >= 2:
        m, s = results[0], results[1]
        logger.info("")
        logger.info("=" * 70)
        logger.info("  COMPARISON (%d-gram KenLM, %d train docs, %d test docs)",
                    args.order, len(train_texts), len(test_texts))
        logger.info("=" * 70)
        logger.info("  %-22s  %-18s  %-18s", "", m["name"], s["name"])
        logger.info("  %-22s  %-18d  %-18d", "Vocab size", m["vocab_size"], s["vocab_size"])
        logger.info("  %-22s  %-18d  %-18d", "Train tokens", m["train_tokens"], s["train_tokens"])
        logger.info("  %-22s  %-18d  %-18d", "Test tokens", m["test_tokens_raw"], s["test_tokens_raw"])
        logger.info("  %-22s  %-18.2f  %-18.2f", "Chars/token", m["chars_per_token"], s["chars_per_token"])
        logger.info("  %-22s  %-18.2f  %-18.2f", "Perplexity", m["ppl"], s["ppl"])
        logger.info("  %-22s  %-18.4f  %-18.4f", "BPC", m["bpc"], s["bpc"])
        logger.info("  %-22s  %-18.2f%%  %-17.2f%%", "OOV rate",
                    m["oov_rate"] * 100, s["oov_rate"] * 100)
        logger.info("")
        bpc_diff = m["bpc"] - s["bpc"]
        if bpc_diff < 0:
            logger.info("  >>> %s wins by %.4f BPC", m["name"], -bpc_diff)
        elif bpc_diff > 0:
            logger.info("  >>> %s wins by %.4f BPC", s["name"], bpc_diff)
        else:
            logger.info("  >>> Tied on BPC")
        logger.info("")
        logger.info("  BPC = bits per character (normalized for tokenization granularity)")
        logger.info("  Lower BPC = more predictable token sequences = better tokenization")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
