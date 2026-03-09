#!/usr/bin/env python3
"""
Compare tokenizer quality via n-gram language model perplexity.

Trains a simple smoothed trigram LM on tokenized text and measures
perplexity on held-out data. Lower perplexity = tokens are more predictable
= better tokenization for language modeling.

Pipeline:
  1. Load raw text, split 90/10 into train/test
  2. Tokenize with Morfessor v4 and SentencePiece
  3. Train smoothed trigram LM on train token IDs
  4. Compute perplexity on test token IDs
  5. Compare

Usage:
    python compare_ngram_ppl.py \
        --input data/sample.parquet \
        --morf-tokenizer ./tokenizer \
        --morf-model data/morfessor/morfessor_telugu.bin \
        --suffix-set data/morfessor/suffix_set.json \
        --sp-model ./sp_tokenizer/sp_telugu.model \
        --num-docs 50000
"""

import math
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from collections import Counter, defaultdict

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
# Smoothed n-gram LM
# ---------------------------------------------------------------------------
class NgramLM:
    """Kneser-Ney-ish smoothed trigram LM.

    Uses simple absolute discounting with backoff:
      P(w | w2, w1) = max(count(w2,w1,w) - d, 0) / count(w2,w1) + lambda * P(w | w1)
      P(w | w1)     = max(count(w1,w) - d, 0) / count(w1) + lambda * P(w)
      P(w)          = count(w) / total
    """

    def __init__(self, discount: float = 0.75):
        self.discount = discount
        # Counts
        self.unigram = Counter()       # w -> count
        self.bigram = Counter()        # (w1, w) -> count
        self.trigram = Counter()       # (w2, w1, w) -> count
        self.bigram_ctx = Counter()    # w1 -> count (bigram context)
        self.trigram_ctx = Counter()   # (w2, w1) -> count (trigram context)
        self.total = 0
        self.vocab_size = 0
        self.BOS = -1  # sentinel for beginning of sequence

    def train(self, token_sequences: list[list[int]]):
        """Train on a list of token ID sequences (one per document)."""
        from tqdm import tqdm

        logger.info("Training trigram LM on %d sequences...", len(token_sequences))

        for seq in tqdm(token_sequences, desc="Counting n-grams", unit=" docs"):
            padded = [self.BOS, self.BOS] + seq
            for i in range(2, len(padded)):
                w = padded[i]
                w1 = padded[i - 1]
                w2 = padded[i - 2]

                self.unigram[w] += 1
                self.total += 1

                self.bigram[(w1, w)] += 1
                self.bigram_ctx[w1] += 1

                self.trigram[(w2, w1, w)] += 1
                self.trigram_ctx[(w2, w1)] += 1

        self.vocab_size = len(self.unigram)
        logger.info("  Vocab: %d types, %d tokens", self.vocab_size, self.total)
        logger.info("  Bigram types: %d, Trigram types: %d",
                    len(self.bigram), len(self.trigram))

    def log_prob(self, w: int, w1: int, w2: int) -> float:
        """Log2 probability of w given context (w2, w1)."""
        d = self.discount

        # Unigram P(w)
        p_uni = (self.unigram.get(w, 0) + 1) / (self.total + self.vocab_size)

        # Bigram P(w|w1)
        bi_ctx = self.bigram_ctx.get(w1, 0)
        if bi_ctx > 0:
            bi_count = self.bigram.get((w1, w), 0)
            # Number of unique continuations from w1
            n_cont_bi = sum(1 for (a, _) in self.bigram if a == w1) if bi_ctx < 100 else max(1, bi_ctx // 10)
            lam_bi = (d * n_cont_bi) / bi_ctx if bi_ctx > 0 else 1.0
            p_bi = max(bi_count - d, 0) / bi_ctx + lam_bi * p_uni
        else:
            p_bi = p_uni

        # Trigram P(w|w2,w1)
        tri_ctx_key = (w2, w1)
        tri_ctx = self.trigram_ctx.get(tri_ctx_key, 0)
        if tri_ctx > 0:
            tri_count = self.trigram.get((w2, w1, w), 0)
            n_cont_tri = sum(1 for (a, b, _) in self.trigram if a == w2 and b == w1) if tri_ctx < 100 else max(1, tri_ctx // 10)
            lam_tri = (d * n_cont_tri) / tri_ctx if tri_ctx > 0 else 1.0
            p_tri = max(tri_count - d, 0) / tri_ctx + lam_tri * p_bi
        else:
            p_tri = p_bi

        # Clamp to avoid log(0)
        p_tri = max(p_tri, 1e-20)
        return math.log2(p_tri)

    def perplexity(self, token_sequences: list[list[int]]) -> tuple[float, int]:
        """Compute perplexity on held-out sequences.

        Returns (perplexity, total_tokens).
        """
        from tqdm import tqdm

        total_log_prob = 0.0
        total_tokens = 0

        for seq in tqdm(token_sequences, desc="Computing PPL", unit=" docs"):
            padded = [self.BOS, self.BOS] + seq
            for i in range(2, len(padded)):
                w = padded[i]
                w1 = padded[i - 1]
                w2 = padded[i - 2]
                total_log_prob += self.log_prob(w, w1, w2)
                total_tokens += 1

        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
        ppl = 2 ** (-avg_log_prob)
        return ppl, total_tokens


# ---------------------------------------------------------------------------
# Precompute bigram/trigram continuation counts for speed
# ---------------------------------------------------------------------------
class FastNgramLM:
    """Faster trigram LM with precomputed continuation counts."""

    def __init__(self, discount: float = 0.75):
        self.discount = discount
        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()
        self.bigram_ctx = Counter()
        self.trigram_ctx = Counter()
        # Precomputed continuation counts
        self.bigram_n_cont = {}   # w1 -> number of unique w following w1
        self.trigram_n_cont = {}  # (w2,w1) -> number of unique w following (w2,w1)
        self.total = 0
        self.vocab_size = 0
        self.BOS = -1

    def train(self, token_sequences: list[list[int]]):
        from tqdm import tqdm

        logger.info("Training trigram LM on %d sequences...", len(token_sequences))

        for seq in tqdm(token_sequences, desc="Counting n-grams", unit=" docs"):
            padded = [self.BOS, self.BOS] + seq
            for i in range(2, len(padded)):
                w = padded[i]
                w1 = padded[i - 1]
                w2 = padded[i - 2]

                self.unigram[w] += 1
                self.total += 1

                self.bigram[(w1, w)] += 1
                self.bigram_ctx[w1] += 1

                self.trigram[(w2, w1, w)] += 1
                self.trigram_ctx[(w2, w1)] += 1

        self.vocab_size = len(self.unigram)

        # Precompute continuation counts
        logger.info("Precomputing continuation counts...")
        bi_cont = defaultdict(set)
        tri_cont = defaultdict(set)
        for (w1, w) in self.bigram:
            bi_cont[w1].add(w)
        for (w2, w1, w) in self.trigram:
            tri_cont[(w2, w1)].add(w)

        self.bigram_n_cont = {k: len(v) for k, v in bi_cont.items()}
        self.trigram_n_cont = {k: len(v) for k, v in tri_cont.items()}

        logger.info("  Vocab: %d types, %d tokens", self.vocab_size, self.total)
        logger.info("  Bigram types: %d, Trigram types: %d",
                    len(self.bigram), len(self.trigram))

    def log_prob(self, w: int, w1: int, w2: int) -> float:
        d = self.discount

        # Unigram (add-1 smoothing)
        p_uni = (self.unigram.get(w, 0) + 1) / (self.total + self.vocab_size)

        # Bigram
        bi_ctx = self.bigram_ctx.get(w1, 0)
        if bi_ctx > 0:
            bi_count = self.bigram.get((w1, w), 0)
            n_cont = self.bigram_n_cont.get(w1, 1)
            lam = (d * n_cont) / bi_ctx
            p_bi = max(bi_count - d, 0) / bi_ctx + lam * p_uni
        else:
            p_bi = p_uni

        # Trigram
        tri_ctx_key = (w2, w1)
        tri_ctx = self.trigram_ctx.get(tri_ctx_key, 0)
        if tri_ctx > 0:
            tri_count = self.trigram.get((w2, w1, w), 0)
            n_cont = self.trigram_n_cont.get(tri_ctx_key, 1)
            lam = (d * n_cont) / tri_ctx
            p_tri = max(tri_count - d, 0) / tri_ctx + lam * p_bi
        else:
            p_tri = p_bi

        p_tri = max(p_tri, 1e-20)
        return math.log2(p_tri)

    def perplexity(self, token_sequences: list[list[int]]) -> tuple[float, int]:
        from tqdm import tqdm

        total_log_prob = 0.0
        total_tokens = 0

        for seq in tqdm(token_sequences, desc="Computing PPL", unit=" docs"):
            padded = [self.BOS, self.BOS] + seq
            for i in range(2, len(padded)):
                total_log_prob += self.log_prob(padded[i], padded[i-1], padded[i-2])
                total_tokens += 1

        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
        ppl = 2 ** (-avg_log_prob)
        return ppl, total_tokens


# ---------------------------------------------------------------------------
# Tokenizer loaders (same as compare_fertility.py)
# ---------------------------------------------------------------------------
def load_morf_encoder(tokenizer_dir: Path, morf_model_path: str = None, suffix_set_path: str = None):
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

    def encode(text: str) -> list[int]:
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare tokenizers via trigram LM perplexity")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--morf-tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--morf-model", type=str, default=None)
    parser.add_argument("--suffix-set", type=str, default=None)
    parser.add_argument("--sp-model", type=str, default=None)
    parser.add_argument("--num-docs", type=int, default=50000)
    parser.add_argument("--test-frac", type=float, default=0.1,
                        help="Fraction of docs for test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    input_path = Path(args.input)

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

    results = []

    # --- Run for each tokenizer ---
    encoders = []
    morf_dir = Path(args.morf_tokenizer)
    if morf_dir.exists():
        enc, vs = load_morf_encoder(morf_dir, args.morf_model, args.suffix_set)
        encoders.append(("Morfessor v4", enc, vs))

    if args.sp_model:
        sp_path = Path(args.sp_model)
        if sp_path.exists():
            enc, vs = load_sp_encoder(sp_path)
            encoders.append(("SentencePiece", enc, vs))

    for name, encode_fn, vocab_size in encoders:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  %s (vocab=%d)", name, vocab_size)
        logger.info("=" * 70)

        # Tokenize
        from tqdm import tqdm
        logger.info("Tokenizing train set...")
        train_seqs = [encode_fn(t) for t in tqdm(train_texts, desc=f"{name} train", unit=" docs")]
        logger.info("Tokenizing test set...")
        test_seqs = [encode_fn(t) for t in tqdm(test_texts, desc=f"{name} test", unit=" docs")]

        train_tokens = sum(len(s) for s in train_seqs)
        test_tokens = sum(len(s) for s in test_seqs)
        total_chars = sum(len(t) for t in train_texts) + sum(len(t) for t in test_texts)
        train_chars = sum(len(t) for t in train_texts)
        test_chars = sum(len(t) for t in test_texts)

        logger.info("  Train: %d tokens (%.2f chars/token)", train_tokens,
                    train_chars / train_tokens if train_tokens else 0)
        logger.info("  Test:  %d tokens (%.2f chars/token)", test_tokens,
                    test_chars / test_tokens if test_tokens else 0)

        # Train trigram LM
        lm = FastNgramLM(discount=0.75)
        lm.train(train_seqs)

        # Evaluate
        ppl, n_eval = lm.perplexity(test_seqs)

        # Also compute bits-per-character (BPC) — the fairest cross-tokenizer metric
        # BPC = total_log2_prob / total_characters (not tokens)
        total_log_prob = 0.0
        for seq in test_seqs:
            padded = [lm.BOS, lm.BOS] + seq
            for i in range(2, len(padded)):
                total_log_prob += lm.log_prob(padded[i], padded[i-1], padded[i-2])
        bpc = -total_log_prob / test_chars if test_chars > 0 else 0

        logger.info("")
        logger.info("  Results:")
        logger.info("    Perplexity:        %.2f", ppl)
        logger.info("    Bits-per-char:     %.4f", bpc)
        logger.info("    Chars/token:       %.2f", test_chars / test_tokens if test_tokens else 0)
        logger.info("    Test tokens:       %d", n_eval)
        logger.info("    Test chars:        %d", test_chars)

        results.append({
            "name": name,
            "vocab_size": vocab_size,
            "perplexity": ppl,
            "bpc": bpc,
            "chars_per_token": test_chars / test_tokens if test_tokens else 0,
            "train_tokens": train_tokens,
            "test_tokens": test_tokens,
            "test_chars": test_chars,
        })

        # Free memory
        del lm, train_seqs, test_seqs

    # --- Comparison ---
    if len(results) >= 2:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  COMPARISON")
        logger.info("=" * 70)
        logger.info("  %-20s  %-15s  %-15s", "", results[0]["name"], results[1]["name"])
        logger.info("  %-20s  %-15d  %-15d", "Vocab size", results[0]["vocab_size"], results[1]["vocab_size"])
        logger.info("  %-20s  %-15.2f  %-15.2f", "Perplexity", results[0]["perplexity"], results[1]["perplexity"])
        logger.info("  %-20s  %-15.4f  %-15.4f", "Bits-per-char (BPC)", results[0]["bpc"], results[1]["bpc"])
        logger.info("  %-20s  %-15.2f  %-15.2f", "Chars/token", results[0]["chars_per_token"], results[1]["chars_per_token"])
        logger.info("  %-20s  %-15d  %-15d", "Test tokens", results[0]["test_tokens"], results[1]["test_tokens"])
        logger.info("")
        bpc_diff = results[0]["bpc"] - results[1]["bpc"]
        if bpc_diff < 0:
            logger.info("  >>> %s wins by %.4f BPC (lower = better)", results[0]["name"], -bpc_diff)
        elif bpc_diff > 0:
            logger.info("  >>> %s wins by %.4f BPC (lower = better)", results[1]["name"], bpc_diff)
        else:
            logger.info("  >>> Tied on BPC")
        logger.info("")
        logger.info("  NOTE: BPC (bits-per-character) is the fairest comparison metric —")
        logger.info("  it normalizes for different tokenization granularity.")
        logger.info("  Perplexity is NOT directly comparable across different vocab sizes.")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
