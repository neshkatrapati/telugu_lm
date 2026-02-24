#!/usr/bin/env python3
"""
Telugu Morpheme Stitcher Model
==============================
Learns to predict word boundaries from a stream of bare morphemes.

Given a sequence of morphemes like: విద్యార్థు ల కు మంచి
The stitcher predicts at each boundary: JOIN (same word) or SPLIT (new word).

Training data comes from Morfessor's @@ segmented output:
  విద్యార్థు@@ ల@@ కు మంచి
  → boundary(విద్యార్థు, ల) = JOIN  (both have @@, same word)
  → boundary(ల, కు) = JOIN  (ల has @@, same word)
  → boundary(కు, మంచి) = SPLIT (కు has no @@, word ended)

Model: character n-gram + morpheme identity features → logistic regression.

Features at each boundary (left_morpheme, right_morpheme):
  - Last N chars of left morpheme (char n-grams, N=1,2,3)
  - First N chars of right morpheme (char n-grams, N=1,2,3)
  - Left morpheme identity (if frequent enough)
  - Right morpheme identity (if frequent enough)
  - Left morpheme length bucket
  - Right morpheme length bucket
  - Bigram: (left_morph, right_morph) if both frequent

Usage:
    # Generate training data and train stitcher
    python train_stitcher.py --seg-file data/sample_raw.seg.txt --output data/stitcher

    # Evaluate only
    python train_stitcher.py --seg-file data/sample_raw.seg.txt --output data/stitcher --eval-only
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telugu detection
# ---------------------------------------------------------------------------
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

def has_telugu(s: str) -> bool:
    return bool(_TELUGU_RE.search(s))

# ---------------------------------------------------------------------------
# Step 1: Extract (left_morph, right_morph, label) from @@ segmented text
# ---------------------------------------------------------------------------
def extract_boundary_examples(seg_file: Path, max_lines: int = 0):
    """Parse @@ segmented file and extract boundary examples.

    Each adjacent pair of morphemes within the token stream produces one example.
    Label: 1 = JOIN (same word), 0 = SPLIT (different words).

    Only considers boundaries where at least one side has Telugu.

    Returns list of (left_base, right_base, label).
    """
    examples = []
    n_join = 0
    n_split = 0

    with open(seg_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Extracting boundaries", unit=" lines")):
            if max_lines and i >= max_lines:
                break

            tokens = line.split()
            if len(tokens) < 2:
                continue

            # Parse tokens into (base, is_continuation) pairs
            parsed = []
            for t in tokens:
                if t.endswith("@@"):
                    parsed.append((t[:-2], True))   # continuation = part of same word
                else:
                    parsed.append((t, False))

            # Generate boundary examples from adjacent pairs
            for j in range(len(parsed) - 1):
                left_base, left_cont = parsed[j]
                right_base, right_cont = parsed[j + 1]

                # Only consider boundaries involving Telugu morphemes
                if not has_telugu(left_base) and not has_telugu(right_base):
                    continue

                # Label: if left token has @@, these are part of same word → JOIN
                label = 1 if left_cont else 0

                examples.append((left_base, right_base, label))
                if label == 1:
                    n_join += 1
                else:
                    n_split += 1

    logger.info("Extracted %d boundary examples: %d JOIN (%.1f%%), %d SPLIT (%.1f%%)",
                len(examples), n_join, 100 * n_join / len(examples),
                n_split, 100 * n_split / len(examples))
    return examples


# ---------------------------------------------------------------------------
# Step 2: Feature extraction
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Extracts character n-gram + morpheme identity features for boundary classification."""

    def __init__(self, morph_min_freq: int = 10, bigram_min_freq: int = 5,
                 char_ngram_sizes: tuple = (1, 2, 3)):
        self.morph_min_freq = morph_min_freq
        self.bigram_min_freq = bigram_min_freq
        self.char_ngram_sizes = char_ngram_sizes

        # Built during fit()
        self.feature_to_idx = {}
        self.frequent_morphs = set()
        self.frequent_bigrams = set()
        self.n_features = 0

    def fit(self, examples: list[tuple[str, str, int]]):
        """Learn feature vocabulary from training examples."""
        logger.info("Fitting feature extractor on %d examples...", len(examples))

        # Count morpheme frequencies
        morph_freq = Counter()
        bigram_freq = Counter()
        char_ngram_freq = Counter()

        for left, right, label in examples:
            morph_freq[left] += 1
            morph_freq[right] += 1
            bigram_freq[(left, right)] += 1

            # Char n-grams at boundary
            for n in self.char_ngram_sizes:
                # Last n chars of left
                suffix = left[-n:] if len(left) >= n else left
                char_ngram_freq[f"L_suffix_{n}={suffix}"] += 1
                # First n chars of right
                prefix = right[:n] if len(right) >= n else right
                char_ngram_freq[f"R_prefix_{n}={prefix}"] += 1

        # Select frequent morphemes
        self.frequent_morphs = {m for m, f in morph_freq.items() if f >= self.morph_min_freq}
        self.frequent_bigrams = {b for b, f in bigram_freq.items()
                                  if f >= self.bigram_min_freq
                                  and b[0] in self.frequent_morphs
                                  and b[1] in self.frequent_morphs}

        logger.info("  Frequent morphemes: %d (min_freq=%d)",
                    len(self.frequent_morphs), self.morph_min_freq)
        logger.info("  Frequent bigrams: %d (min_freq=%d)",
                    len(self.frequent_bigrams), self.bigram_min_freq)

        # Build feature index
        feat_idx = {}
        idx = 0

        # Char n-gram features (only those appearing 2+ times)
        for feat, freq in sorted(char_ngram_freq.items()):
            if freq >= 2:
                feat_idx[feat] = idx
                idx += 1

        # Morpheme identity features
        for m in sorted(self.frequent_morphs):
            feat_idx[f"L_morph={m}"] = idx
            idx += 1
            feat_idx[f"R_morph={m}"] = idx
            idx += 1

        # Bigram features
        for (l, r) in sorted(self.frequent_bigrams):
            feat_idx[f"bigram={l}|{r}"] = idx
            idx += 1

        # Length bucket features
        for side in ("L", "R"):
            for bucket in range(1, 11):  # 1-10+ chars
                feat_idx[f"{side}_len={bucket}"] = idx
                idx += 1

        # Telugu-on-each-side features
        feat_idx["L_telugu=1"] = idx; idx += 1
        feat_idx["L_telugu=0"] = idx; idx += 1
        feat_idx["R_telugu=1"] = idx; idx += 1
        feat_idx["R_telugu=0"] = idx; idx += 1

        self.feature_to_idx = feat_idx
        self.n_features = idx
        logger.info("  Total features: %d", self.n_features)

    def transform_one(self, left: str, right: str) -> list[int]:
        """Extract feature indices for a single boundary."""
        active = []

        # Char n-gram features
        for n in self.char_ngram_sizes:
            suffix = left[-n:] if len(left) >= n else left
            key = f"L_suffix_{n}={suffix}"
            if key in self.feature_to_idx:
                active.append(self.feature_to_idx[key])

            prefix = right[:n] if len(right) >= n else right
            key = f"R_prefix_{n}={prefix}"
            if key in self.feature_to_idx:
                active.append(self.feature_to_idx[key])

        # Morpheme identity
        if left in self.frequent_morphs:
            key = f"L_morph={left}"
            if key in self.feature_to_idx:
                active.append(self.feature_to_idx[key])

        if right in self.frequent_morphs:
            key = f"R_morph={right}"
            if key in self.feature_to_idx:
                active.append(self.feature_to_idx[key])

        # Bigram
        if (left, right) in self.frequent_bigrams:
            key = f"bigram={left}|{right}"
            if key in self.feature_to_idx:
                active.append(self.feature_to_idx[key])

        # Length buckets
        l_len = min(len(left), 10)
        r_len = min(len(right), 10)
        key_l = f"L_len={l_len}"
        key_r = f"R_len={r_len}"
        if key_l in self.feature_to_idx:
            active.append(self.feature_to_idx[key_l])
        if key_r in self.feature_to_idx:
            active.append(self.feature_to_idx[key_r])

        # Telugu flags
        l_tel = "1" if has_telugu(left) else "0"
        r_tel = "1" if has_telugu(right) else "0"
        key = f"L_telugu={l_tel}"
        if key in self.feature_to_idx:
            active.append(self.feature_to_idx[key])
        key = f"R_telugu={r_tel}"
        if key in self.feature_to_idx:
            active.append(self.feature_to_idx[key])

        return active

    def transform_batch(self, examples: list[tuple[str, str, int]]):
        """Transform examples into sparse feature matrix + labels.

        Returns (row_indices, col_indices, labels) for sparse matrix construction.
        """
        rows = []
        cols = []
        labels = []

        for i, (left, right, label) in enumerate(examples):
            active = self.transform_one(left, right)
            for col in active:
                rows.append(i)
                cols.append(col)
            labels.append(label)

        return rows, cols, np.array(labels, dtype=np.float32)

    def save(self, path: Path):
        """Save feature extractor state."""
        state = {
            "morph_min_freq": self.morph_min_freq,
            "bigram_min_freq": self.bigram_min_freq,
            "char_ngram_sizes": self.char_ngram_sizes,
            "feature_to_idx": self.feature_to_idx,
            "frequent_morphs": list(self.frequent_morphs),
            "frequent_bigrams": [(l, r) for l, r in self.frequent_bigrams],
            "n_features": self.n_features,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved feature extractor to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureExtractor":
        """Load feature extractor state."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        fe = cls(
            morph_min_freq=state["morph_min_freq"],
            bigram_min_freq=state["bigram_min_freq"],
            char_ngram_sizes=tuple(state["char_ngram_sizes"]),
        )
        fe.feature_to_idx = state["feature_to_idx"]
        fe.frequent_morphs = set(state["frequent_morphs"])
        fe.frequent_bigrams = {(l, r) for l, r in state["frequent_bigrams"]}
        fe.n_features = state["n_features"]
        return fe


# ---------------------------------------------------------------------------
# Step 3: Train logistic regression stitcher
# ---------------------------------------------------------------------------
class StitcherModel:
    """Logistic regression for morpheme boundary classification.

    Uses sparse features and SGD for memory efficiency on large datasets.
    """

    def __init__(self, n_features: int, lr: float = 0.1, reg: float = 1e-4):
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.lr = lr
        self.reg = reg
        self.n_features = n_features

    def predict_proba(self, active_features: list[int]) -> float:
        """Predict P(JOIN) for a single example."""
        logit = self.bias + sum(self.weights[f] for f in active_features)
        # Clip for numerical stability
        logit = max(-30.0, min(30.0, logit))
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, active_features: list[int], threshold: float = 0.5) -> int:
        """Predict JOIN (1) or SPLIT (0)."""
        return 1 if self.predict_proba(active_features) >= threshold else 0

    def train_epoch(self, examples: list[tuple[str, str, int]],
                    feature_extractor: FeatureExtractor,
                    shuffle: bool = True) -> float:
        """One pass of SGD over examples. Returns average loss."""
        indices = list(range(len(examples)))
        if shuffle:
            np.random.shuffle(indices)

        total_loss = 0.0
        for idx in indices:
            left, right, label = examples[idx]
            active = feature_extractor.transform_one(left, right)

            # Forward
            p = self.predict_proba(active)
            # Binary cross-entropy
            eps = 1e-7
            loss = -(label * np.log(p + eps) + (1 - label) * np.log(1 - p + eps))
            total_loss += loss

            # Gradient: d_loss/d_logit = p - label
            grad = p - label

            # Update
            self.bias -= self.lr * grad
            for f in active:
                self.weights[f] -= self.lr * (grad + self.reg * self.weights[f])

        return total_loss / len(examples)

    def evaluate(self, examples: list[tuple[str, str, int]],
                 feature_extractor: FeatureExtractor) -> dict:
        """Evaluate on examples. Returns accuracy, precision, recall, F1 for both classes."""
        tp = fp = tn = fn = 0

        for left, right, label in examples:
            active = feature_extractor.transform_one(left, right)
            pred = self.predict(active)

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0

        # JOIN metrics
        join_precision = tp / (tp + fp) if (tp + fp) else 0
        join_recall = tp / (tp + fn) if (tp + fn) else 0
        join_f1 = 2 * join_precision * join_recall / (join_precision + join_recall) if (join_precision + join_recall) else 0

        # SPLIT metrics
        split_precision = tn / (tn + fn) if (tn + fn) else 0
        split_recall = tn / (tn + fp) if (tn + fp) else 0
        split_f1 = 2 * split_precision * split_recall / (split_precision + split_recall) if (split_precision + split_recall) else 0

        return {
            "accuracy": accuracy,
            "total": total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "join_precision": join_precision,
            "join_recall": join_recall,
            "join_f1": join_f1,
            "split_precision": split_precision,
            "split_recall": split_recall,
            "split_f1": split_f1,
        }

    def save(self, path: Path):
        """Save model weights."""
        state = {
            "weights": self.weights,
            "bias": self.bias,
            "n_features": self.n_features,
        }
        np.savez(path, **state)
        logger.info("Saved stitcher model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "StitcherModel":
        """Load model weights."""
        data = np.load(path, allow_pickle=True)
        model = cls(n_features=int(data["n_features"]))
        model.weights = data["weights"]
        model.bias = float(data["bias"])
        return model


# ---------------------------------------------------------------------------
# Step 4: Demo stitching
# ---------------------------------------------------------------------------
def demo_stitch(morphemes: list[str], model: StitcherModel,
                feature_extractor: FeatureExtractor) -> str:
    """Stitch a sequence of bare morphemes back into words."""
    if not morphemes:
        return ""

    result = [morphemes[0]]
    for i in range(1, len(morphemes)):
        active = feature_extractor.transform_one(morphemes[i - 1], morphemes[i])
        pred = model.predict(active)
        if pred == 0:  # SPLIT — new word
            result.append(" ")
        result.append(morphemes[i])

    return "".join(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train morpheme stitcher model")
    parser.add_argument("--seg-file", type=str, required=True,
                        help="Path to @@ segmented corpus file (.seg.txt)")
    parser.add_argument("--output", type=str, default="data/stitcher",
                        help="Output directory for model files")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="Max lines to read (0 = all)")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fraction of data for test (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--morph-min-freq", type=int, default=10,
                        help="Min freq for morpheme identity features (default: 10)")
    parser.add_argument("--bigram-min-freq", type=int, default=5,
                        help="Min freq for bigram features (default: 5)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load existing model and evaluate")
    parser.add_argument("--demo", type=str, nargs="*", default=None,
                        help="Demo: space-separated morphemes to stitch")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fe_path = output_dir / "feature_extractor.pkl"
    model_path = output_dir / "stitcher_model.npz"

    # --- Demo mode ---
    if args.demo is not None:
        if not fe_path.exists() or not model_path.exists():
            logger.error("No saved model found in %s", output_dir)
            sys.exit(1)
        fe = FeatureExtractor.load(fe_path)
        model = StitcherModel.load(model_path)
        morphemes = args.demo
        result = demo_stitch(morphemes, model, fe)
        print(f"Morphemes: {' | '.join(morphemes)}")
        print(f"Stitched:  {result}")
        return

    # --- Extract examples ---
    seg_file = Path(args.seg_file)
    if not seg_file.exists():
        logger.error("Segmented file not found: %s", seg_file)
        sys.exit(1)

    logger.info("Extracting boundary examples from %s", seg_file)
    examples = extract_boundary_examples(seg_file, max_lines=args.max_lines)

    # --- Train/test split ---
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    test_size = int(len(examples) * args.test_split)
    test_idx = set(indices[:test_size])

    train_examples = [examples[i] for i in range(len(examples)) if i not in test_idx]
    test_examples = [examples[i] for i in range(len(examples)) if i in test_idx]

    logger.info("Train: %d examples, Test: %d examples", len(train_examples), len(test_examples))

    if args.eval_only:
        if not fe_path.exists() or not model_path.exists():
            logger.error("No saved model found in %s", output_dir)
            sys.exit(1)
        fe = FeatureExtractor.load(fe_path)
        model = StitcherModel.load(model_path)
    else:
        # --- Fit feature extractor ---
        fe = FeatureExtractor(
            morph_min_freq=args.morph_min_freq,
            bigram_min_freq=args.bigram_min_freq,
        )
        fe.fit(train_examples)
        fe.save(fe_path)

        # --- Train ---
        model = StitcherModel(n_features=fe.n_features, lr=args.lr)

        for epoch in range(args.epochs):
            loss = model.train_epoch(train_examples, fe)
            # Quick eval on test
            metrics = model.evaluate(test_examples, fe)
            logger.info("Epoch %d/%d — loss: %.4f, test acc: %.4f, "
                        "JOIN F1: %.4f, SPLIT F1: %.4f",
                        epoch + 1, args.epochs, loss, metrics["accuracy"],
                        metrics["join_f1"], metrics["split_f1"])

        model.save(model_path)

    # --- Final evaluation ---
    logger.info("=" * 60)
    logger.info("Final evaluation on test set:")
    metrics = model.evaluate(test_examples, fe)
    logger.info("  Accuracy:        %.4f (%d/%d)", metrics["accuracy"],
                metrics["tp"] + metrics["tn"], metrics["total"])
    logger.info("  JOIN  — P: %.4f  R: %.4f  F1: %.4f",
                metrics["join_precision"], metrics["join_recall"], metrics["join_f1"])
    logger.info("  SPLIT — P: %.4f  R: %.4f  F1: %.4f",
                metrics["split_precision"], metrics["split_recall"], metrics["split_f1"])
    logger.info("  Confusion: TP=%d FP=%d TN=%d FN=%d",
                metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"])

    # --- Demo on some examples ---
    logger.info("=" * 60)
    logger.info("Demo stitching:")

    # Extract a few word sequences from test data for demo
    demo_seqs = _extract_demo_sequences(seg_file, n=10)
    for morphemes, original in demo_seqs:
        stitched = demo_stitch(morphemes, model, fe)
        match = "OK" if stitched == original else "MISMATCH"
        logger.info("  Original:  %s", original)
        logger.info("  Morphemes: %s", " | ".join(morphemes))
        logger.info("  Stitched:  %s  [%s]", stitched, match)
        logger.info("")


def _extract_demo_sequences(seg_file: Path, n: int = 10) -> list[tuple[list[str], str]]:
    """Extract n demo sequences: (bare_morphemes, original_text)."""
    demos = []
    with open(seg_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200:  # sample from first 200 lines
                break
            tokens = line.split()
            if len(tokens) < 4:
                continue

            # Reconstruct original from @@ format
            morphemes = []
            original_words = []
            current_word = []
            for t in tokens:
                if t.endswith("@@"):
                    base = t[:-2]
                    morphemes.append(base)
                    current_word.append(base)
                else:
                    morphemes.append(t)
                    current_word.append(t)
                    original_words.append("".join(current_word))
                    current_word = []
            if current_word:
                original_words.append("".join(current_word))

            original = " ".join(original_words)
            if has_telugu(original) and len(morphemes) >= 3:
                demos.append((morphemes, original))
                if len(demos) >= n:
                    break

    return demos


if __name__ == "__main__":
    main()
