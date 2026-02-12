#!/usr/bin/env python3
"""
Telugu LLaMA — Interactive Inference
=====================================
Full pipeline: raw text → Morfessor segmentation → tokenize → model → detokenize → text

Usage:
    python inference.py --checkpoint ./checkpoints/best.pt
    python inference.py --checkpoint ./checkpoints/best.pt --max-tokens 300 --temperature 0.9
    python inference.py --checkpoint ./checkpoints/best.pt --prompt "తెలుగు భాష"
"""

import sys
import re
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


def load_morfessor_model(model_path: Path):
    """Load the Morfessor model for segmentation."""
    import morfessor
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    logger.info("Loaded Morfessor model from %s", model_path)
    return model


def segment_text(text: str, morf_model, separator: str = "@@") -> str:
    """Segment raw text using Morfessor.

    Telugu words → morpheme segments with @@ boundaries
    Non-Telugu words → kept as-is
    """
    tokens = text.split()
    seg_tokens = []

    for token in tokens:
        if TELUGU_WORD_RE.fullmatch(token):
            # Pure Telugu word — segment with Morfessor
            segments = morf_model.viterbi_segment(token)[0]
            for i, seg in enumerate(segments):
                if i < len(segments) - 1:
                    seg_tokens.append(seg + separator)
                else:
                    seg_tokens.append(seg)
        elif TELUGU_WORD_RE.search(token):
            # Mixed token — split on Telugu boundaries
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            for part in parts:
                if not part:
                    continue
                if TELUGU_WORD_RE.fullmatch(part):
                    segments = morf_model.viterbi_segment(part)[0]
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            seg_tokens.append(seg + separator)
                        else:
                            seg_tokens.append(seg)
                else:
                    seg_tokens.append(part)
        else:
            seg_tokens.append(token)

    return " ".join(seg_tokens)


def load_model(checkpoint_path: Path, device: str):
    """Load trained model from checkpoint."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from train_gpt import GPTConfig, build_model

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])

    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    step = checkpoint.get("step", "?")
    val_loss = checkpoint.get("val_loss", "?")
    epoch = checkpoint.get("epoch", "?")
    logger.info("Loaded model from %s (step=%s, epoch=%s, val_loss=%s)", checkpoint_path, step, epoch, val_loss)
    logger.info("  Config: %d layers, %d heads, %d dim, vocab=%d", config.n_layer, config.n_head, config.n_embd, config.vocab_size)

    return model, config


@torch.no_grad()
def generate(model, token_ids: list[int], max_new_tokens: int, temperature: float, top_k: int, device: str) -> list[int]:
    """Generate token IDs autoregressively."""
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Stop on EOS
        if idx_next.item() == 3:  # <eos>
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx[0].tolist()


def desegment(text: str, morf_model, separator: str = "@@") -> str:
    """Reconstruct natural text from space-separated morphemes.

    The tokenizer stores bare morphemes (no @@), so at decode time we can't
    tell which morphemes belong to the same word. We use Morfessor to
    reconstruct: greedily join consecutive Telugu morphemes and check if
    Morfessor would segment the joined form back into those same morphemes.
    If yes, they're one word. If not, they're separate words.
    """
    if not text:
        return text

    TELUGU_CHAR = re.compile(r"[\u0C00-\u0C7F]")
    tokens = text.split()
    if not tokens:
        return text

    # Group consecutive Telugu tokens, keep non-Telugu tokens as-is
    groups = []  # each group is (is_telugu, [tokens])
    for token in tokens:
        is_telugu = bool(TELUGU_CHAR.search(token))
        if groups and groups[-1][0] == is_telugu and is_telugu:
            groups[-1][1].append(token)
        else:
            groups.append((is_telugu, [token]))

    # For each group of Telugu morphemes, try to reconstruct words
    result_parts = []
    for is_telugu, group_tokens in groups:
        if not is_telugu:
            result_parts.append(" ".join(group_tokens))
            continue

        # Greedily build words from morphemes using Morfessor validation
        words = []
        i = 0
        while i < len(group_tokens):
            # Try joining morphemes starting from position i
            # Start with longest possible, shrink until Morfessor agrees
            best_end = i + 1  # at minimum, one morpheme = one word
            for j in range(min(i + 8, len(group_tokens)), i, -1):
                candidate = "".join(group_tokens[i:j])
                segments = morf_model.viterbi_segment(candidate)[0]
                if segments == group_tokens[i:j]:
                    best_end = j
                    break
            words.append("".join(group_tokens[i:best_end]))
            i = best_end
        result_parts.append(" ".join(words))

    return " ".join(result_parts)


def run_inference(
    prompt: str,
    model,
    tokenizer,
    morf_model,
    device: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    separator: str = "@@",
    verbose: bool = False,
):
    """Full inference pipeline: text → segment → tokenize → generate → decode → desegment → text."""

    # Step 1: Morfessor segmentation
    segmented = segment_text(prompt, morf_model, separator)
    if verbose:
        print(f"  [Segmented]  {segmented}")

    # Step 2: Tokenize
    token_ids = tokenizer.encode(segmented, add_bos=True, add_eos=False)
    if verbose:
        print(f"  [Token IDs]  {token_ids[:20]}{'...' if len(token_ids) > 20 else ''} ({len(token_ids)} tokens)")

    # Step 3: Generate
    output_ids = generate(model, token_ids, max_tokens, temperature, top_k, device)

    # Step 4: Decode token IDs → morphemes
    raw_text = tokenizer.decode(output_ids)
    raw_generated = tokenizer.decode(output_ids[len(token_ids):])

    if verbose:
        print(f"  [Raw decode] {raw_generated[:200]}")

    # Step 5: Desegment — merge morphemes back into words using Morfessor
    generated_text = desegment(raw_text, morf_model, separator)
    generated_only = desegment(raw_generated, morf_model, separator)

    if verbose:
        print(f"  [Generated]  {len(output_ids) - len(token_ids)} new tokens")

    return generated_text, generated_only


def main():
    parser = argparse.ArgumentParser(
        description="Telugu LLaMA — Interactive Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checkpoint ./checkpoints/best.pt
  %(prog)s --checkpoint ./checkpoints/best.pt --prompt "తెలుగు భాష"
  %(prog)s --checkpoint ./checkpoints/best.pt --max-tokens 300 --temperature 0.9
  %(prog)s --checkpoint ./checkpoints/best.pt --verbose
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory (default: ./tokenizer)")
    parser.add_argument("--morfessor-model", type=str, default="./data/morfessor/morfessor_telugu.bin", help="Morfessor model path")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--verbose", action="store_true", help="Show intermediate steps (segmentation, token IDs)")
    parser.add_argument("--separator", type=str, default="@@", help="Morfessor separator (default: @@)")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load everything
    logger.info("Loading components...")
    morf_model = load_morfessor_model(Path(args.morfessor_model))

    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer
    tokenizer = MorfessorTokenizer(Path(args.tokenizer))

    model, config = load_model(Path(args.checkpoint), device)

    logger.info("Ready!\n")

    if args.prompt:
        # Single prompt mode
        full_text, generated = run_inference(
            args.prompt, model, tokenizer, morf_model, device,
            args.max_tokens, args.temperature, args.top_k,
            args.separator, args.verbose,
        )
        print(f"\nPrompt:    {args.prompt}")
        print(f"Generated: {generated}")
        print(f"Full:      {full_text}")
    else:
        # Interactive mode
        print("=" * 60)
        print("Telugu LLaMA — Interactive Generation")
        print("Type your prompt and press Enter. Type 'quit' to exit.")
        print("=" * 60)

        while True:
            try:
                prompt = input("\n>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not prompt or prompt.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            full_text, generated = run_inference(
                prompt, model, tokenizer, morf_model, device,
                args.max_tokens, args.temperature, args.top_k,
                args.separator, args.verbose,
            )
            print(f"\n{full_text}")


if __name__ == "__main__":
    main()
