#!/usr/bin/env python3
"""
Bits-Per-Character (BPC) Evaluation
====================================
Compare two language models fairly using BPC — a tokenizer-agnostic metric.

BPC = (total_nll_in_nats × log₂(e)) / total_characters

This normalizes away vocabulary size and tokenization granularity differences,
making it fair to compare a 32K Morfessor model against a 48K SentencePiece model.

Usage:
    # Compare two HuggingFace models on a Telugu text file
    python eval_bpc.py \\
        --model-a dvitvaai/pothana-v1-morfessor \\
        --model-b dvitvaai/pothana-base-300M \\
        --eval-text eval_corpus.txt

    # With labels for cleaner output
    python eval_bpc.py \\
        --model-a dvitvaai/pothana-v1-morfessor --label-a "Morfessor v1" \\
        --model-b dvitvaai/pothana-base-300M    --label-b "SP v2" \\
        --eval-text eval_corpus.txt

    # Use a parquet file instead
    python eval_bpc.py \\
        --model-a dvitvaai/pothana-v1-morfessor \\
        --model-b dvitvaai/pothana-base-300M \\
        --eval-parquet data.parquet --text-column text --max-samples 500

    # Single model eval
    python eval_bpc.py \\
        --model-a dvitvaai/pothana-base-300M \\
        --eval-text eval_corpus.txt
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG2_E = math.log2(math.e)  # ≈ 1.4427


def load_eval_texts(args):
    """Load evaluation texts from file or parquet. Returns list of strings."""
    texts = []

    if args.eval_text:
        path = Path(args.eval_text)
        if not path.exists():
            logger.error("Eval text file not found: %s", path)
            sys.exit(1)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and len(line) >= args.min_chars:
                    texts.append(line)
        logger.info("Loaded %d lines from %s", len(texts), path)

    elif args.eval_parquet:
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas required for parquet support: pip install pandas pyarrow")
            sys.exit(1)

        path = Path(args.eval_parquet)
        if not path.exists():
            logger.error("Parquet file not found: %s", path)
            sys.exit(1)

        df = pd.read_parquet(path)
        col = args.text_column
        if col not in df.columns:
            logger.error("Column '%s' not found. Available: %s", col, list(df.columns))
            sys.exit(1)

        for text in df[col].dropna():
            text = str(text).strip()
            if text and len(text) >= args.min_chars:
                texts.append(text)
        logger.info("Loaded %d texts from %s (column: %s)", len(texts), path, col)

    else:
        logger.error("Must provide --eval-text or --eval-parquet")
        sys.exit(1)

    if args.max_samples and len(texts) > args.max_samples:
        # Deterministic subset
        rng = np.random.RandomState(42)
        indices = rng.choice(len(texts), args.max_samples, replace=False)
        texts = [texts[i] for i in sorted(indices)]
        logger.info("Subsampled to %d texts", len(texts))

    total_chars = sum(len(t) for t in texts)
    logger.info("Total characters: %d (%.1fK)", total_chars, total_chars / 1000)

    return texts


def compute_bpc(model_id, texts, device, max_length, batch_size, label=None):
    """Compute BPC for a single model over the given texts.

    For each text:
      1. Tokenize with the model's tokenizer
      2. Slide a window of max_length tokens, computing NLL at each position
      3. Sum up total NLL (in nats) across all tokens
      4. Count raw characters in the original text

    BPC = (total_nll_nats × log₂(e)) / total_characters
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = label or model_id
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluating: %s", name)
    logger.info("=" * 60)

    # Load model and tokenizer
    t0 = time.time()
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    load_time = time.time() - t0
    logger.info("Loaded in %.1fs (%.1fM params)", load_time,
                sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    total_nll_nats = 0.0
    total_tokens = 0
    total_chars = 0
    n_texts = 0

    t0 = time.time()

    for i, text in enumerate(texts):
        n_chars = len(text)
        if n_chars == 0:
            continue

        # Tokenize — let the tokenizer handle BOS/EOS
        encoding = tokenizer(text, return_tensors="pt", truncation=False,
                             add_special_tokens=True)
        input_ids = encoding["input_ids"][0]  # (seq_len,)
        seq_len = input_ids.size(0)

        if seq_len < 2:
            continue  # need at least 2 tokens for 1 prediction

        # Sliding window for sequences longer than max_length
        # Accumulate NLL for every token position (except the first)
        text_nll = 0.0
        text_tokens = 0

        stride = max_length // 2  # 50% overlap for better context
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_length, seq_len)
            chunk_ids = input_ids[begin:end].unsqueeze(0).to(device)

            # Target: shift by 1
            target_ids = chunk_ids.clone()
            # Mask out tokens in the overlap region that were already scored
            # Only score tokens from max(1, begin) onward relative to chunk start
            if begin > 0:
                overlap = begin + max_length - end  # how many tokens overlap
                # Actually simpler: in the overlap region, mask targets
                # so we don't double-count
                n_already_scored = stride  # tokens before the new stride portion
                # But first chunk starts scoring from position 1
                # Subsequent chunks: only score the stride portion at the end
                target_ids[0, :max(0, end - begin - stride)] = -100
            else:
                # First chunk: don't score position 0 (no context for it)
                target_ids[0, 0] = -100

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(chunk_ids, labels=target_ids)
                # outputs.loss is mean over non-masked tokens
                # We need the SUM of NLLs
                n_scored = (target_ids != -100).sum().item()
                if n_scored > 0:
                    text_nll += outputs.loss.item() * n_scored
                    text_tokens += n_scored

            if end >= seq_len:
                break

        total_nll_nats += text_nll
        total_tokens += text_tokens
        total_chars += n_chars
        n_texts += 1

        # Progress logging
        if (i + 1) % 100 == 0 or i == len(texts) - 1:
            elapsed = time.time() - t0
            running_bpc = (total_nll_nats * LOG2_E) / total_chars if total_chars > 0 else 0
            logger.info("  [%d/%d] chars=%dk tokens=%dk running_bpc=%.4f (%.1f texts/s)",
                        i + 1, len(texts), total_chars // 1000, total_tokens // 1000,
                        running_bpc, n_texts / elapsed)

    eval_time = time.time() - t0

    # Compute final metrics
    bpc = (total_nll_nats * LOG2_E) / total_chars if total_chars > 0 else float("inf")
    avg_nll = total_nll_nats / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_nll) if avg_nll < 20 else float("inf")  # avoid overflow
    tok_per_char = total_tokens / total_chars if total_chars > 0 else 0

    results = {
        "model": model_id,
        "label": name,
        "bpc": bpc,
        "perplexity": ppl,
        "avg_nll_nats": avg_nll,
        "total_nll_nats": total_nll_nats,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": tok_per_char,
        "n_texts": n_texts,
        "eval_time_s": eval_time,
        "vocab_size": tokenizer.vocab_size,
    }

    logger.info("")
    logger.info("Results for: %s", name)
    logger.info("  BPC:              %.4f", bpc)
    logger.info("  Perplexity:       %.2f", ppl)
    logger.info("  Avg NLL (nats):   %.4f", avg_nll)
    logger.info("  Tokens/char:      %.3f", tok_per_char)
    logger.info("  Total tokens:     %d", total_tokens)
    logger.info("  Total chars:      %d", total_chars)
    logger.info("  Texts evaluated:  %d", n_texts)
    logger.info("  Eval time:        %.1fs", eval_time)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def print_comparison(results_a, results_b):
    """Print a side-by-side comparison table."""
    a = results_a
    b = results_b

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    logger.info("")

    header = f"{'Metric':<22} {'[A] ' + a['label']:<25} {'[B] ' + b['label']:<25} {'Δ':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    def row(metric, va, vb, fmt=".4f", lower_better=True):
        delta = vb - va
        arrow = "◀ A" if (delta > 0 and lower_better) or (delta < 0 and not lower_better) else "◀ B"
        if abs(delta) < 1e-6:
            arrow = "  ="
        logger.info(f"{metric:<22} {va:<25{fmt}} {vb:<25{fmt}} {delta:>+8{fmt}} {arrow}")

    row("BPC", a["bpc"], b["bpc"])
    row("Perplexity", a["perplexity"], b["perplexity"], fmt=".2f")
    row("Avg NLL (nats)", a["avg_nll_nats"], b["avg_nll_nats"])
    row("Tokens/char", a["tokens_per_char"], b["tokens_per_char"], fmt=".3f")
    row("Vocab size", a["vocab_size"], b["vocab_size"], fmt="d", lower_better=False)

    logger.info("")
    bpc_diff = b["bpc"] - a["bpc"]
    pct = (bpc_diff / a["bpc"]) * 100 if a["bpc"] > 0 else 0
    if abs(bpc_diff) < 0.001:
        logger.info("Models are virtually identical in BPC.")
    else:
        winner = a["label"] if bpc_diff > 0 else b["label"]
        logger.info("%s is better by %.4f BPC (%.1f%% relative improvement)", winner, abs(bpc_diff), abs(pct))


def main():
    parser = argparse.ArgumentParser(
        description="Compare language models using Bits-Per-Character (BPC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python eval_bpc.py \\
      --model-a dvitvaai/pothana-v1 --label-a "Morfessor v1" \\
      --model-b dvitvaai/pothana-base-300M --label-b "SP v2" \\
      --eval-text eval_corpus.txt

  # Single model
  python eval_bpc.py --model-a dvitvaai/pothana-base-300M --eval-text eval.txt
        """,
    )

    # Models
    parser.add_argument("--model-a", type=str, required=True, help="HF model ID or local path (model A)")
    parser.add_argument("--model-b", type=str, default=None, help="HF model ID or local path (model B, optional)")
    parser.add_argument("--label-a", type=str, default=None, help="Display label for model A")
    parser.add_argument("--label-b", type=str, default=None, help="Display label for model B")

    # Data
    parser.add_argument("--eval-text", type=str, default=None,
                        help="Path to evaluation text file (one document/paragraph per line)")
    parser.add_argument("--eval-parquet", type=str, default=None,
                        help="Path to parquet file for evaluation")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Column name in parquet file (default: text)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of texts to evaluate (for speed)")
    parser.add_argument("--min-chars", type=int, default=20,
                        help="Minimum characters per text (default: 20)")

    # Eval settings
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length for sliding window (default: 2048)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # Load eval texts
    texts = load_eval_texts(args)
    if not texts:
        logger.error("No evaluation texts found!")
        sys.exit(1)

    # Evaluate model A
    results_a = compute_bpc(
        args.model_a, texts, device, args.max_length, batch_size=1,
        label=args.label_a,
    )

    # Evaluate model B (if provided)
    results_b = None
    if args.model_b:
        results_b = compute_bpc(
            args.model_b, texts, device, args.max_length, batch_size=1,
            label=args.label_b,
        )

    # Comparison
    if results_b:
        print_comparison(results_a, results_b)
    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL: %s — BPC = %.4f", results_a["label"], results_a["bpc"])
        logger.info("=" * 60)

    # Save results
    if args.output_json:
        import json
        output = {"model_a": results_a}
        if results_b:
            output["model_b"] = results_b
            output["bpc_diff"] = results_b["bpc"] - results_a["bpc"]
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
