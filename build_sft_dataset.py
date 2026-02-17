#!/usr/bin/env python3
"""
Build SFT Dataset for Pothana Base 300M
=========================================
Converts translated multi-turn Telugu conversations into tokenized training
sequences with chat template formatting and loss masking.

Input:  translated.jsonl — full conversations from Gemini
Output: train.bin, val.bin, test.bin (uint32 token IDs)
        train.mask.bin, val.mask.bin, test.mask.bin (uint8 loss masks)
        meta.json (dataset statistics)

Chat Template (with special tokens):
    <bos> <|system|> {sys} <|end|> <|user|> {user} <|end|> <|assistant|> {asst} <|end|> ... <eos>

Loss is computed ONLY on assistant response tokens (mask=1).
Everything else (system, user, context, role markers) is masked (mask=0).

Windowed expansion: each conversation with N turn pairs produces N training
examples, each with a sliding context window of the last K turns.

Usage:
    python build_sft_dataset.py \\
        --input ./data/translated/translated.jsonl \\
        --tokenizer ./tokenizer \\
        --morfessor ./morfessor_telugu.bin \\
        --output ./sft_data \\
        --context-window 4 \\
        --max-seq-len 2048

    # Dry run — just print stats, don't write files
    python build_sft_dataset.py \\
        --input ./data/translated/translated.jsonl \\
        --tokenizer ./tokenizer \\
        --morfessor ./morfessor_telugu.bin \\
        --output ./sft_data \\
        --dry-run
"""

import os
import re
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Special Token IDs
# ============================================================================
# New chat special tokens are appended AFTER the existing vocab.
# IDs are assigned dynamically at runtime based on tokenizer.vocab_size.
# e.g. if vocab_size=42000, then <|system|>=42000, <|user|>=42001, etc.
#
# This avoids collisions with existing token IDs.
# The embedding layer must be resized: new_vocab = old_vocab + 4
NEW_SPECIAL_TOKEN_NAMES = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]

# Existing special tokens from base tokenizer
BOS_ID = 2
EOS_ID = 3

TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


# ============================================================================
# Morfessor Segmentation (from inference.py)
# ============================================================================
def load_morfessor_model(model_path: Path):
    """Load the Morfessor model for segmentation."""
    import morfessor
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(str(model_path))
    logger.info("Loaded Morfessor model from %s", model_path)
    return model


def segment_text(text: str, morf_model, separator: str = "@@") -> str:
    """Segment raw text using Morfessor with @@ continuation markers.

    - Pure Telugu words -> Morfessor morpheme segments with @@ boundaries
    - Pure non-Telugu words -> kept as-is
    - Mixed-script tokens (e.g. "2024లో") -> split at script boundary with @@
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
            # Mixed-script token — split at Telugu/non-Telugu boundaries
            parts = re.split(r"([\u0C00-\u0C7F]+)", token)
            parts = [p for p in parts if p]

            for part_idx, part in enumerate(parts):
                is_last_part = (part_idx == len(parts) - 1)

                if TELUGU_WORD_RE.fullmatch(part):
                    segments = morf_model.viterbi_segment(part)[0]
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            seg_tokens.append(seg + separator)
                        else:
                            if not is_last_part:
                                seg_tokens.append(seg + separator)
                            else:
                                seg_tokens.append(seg)
                else:
                    if not is_last_part:
                        seg_tokens.append(part + separator)
                    else:
                        seg_tokens.append(part)
        else:
            # Pure non-Telugu word — keep as-is
            seg_tokens.append(token)

    return " ".join(seg_tokens)


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class SFTExample:
    """A single SFT training example with token IDs and loss mask."""
    input_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)  # 1 = compute loss, 0 = ignore

    @property
    def length(self):
        return len(self.input_ids)


# ============================================================================
# Chat Template Builder
# ============================================================================
class ChatTemplateBuilder:
    """Builds tokenized SFT sequences from conversations using chat template."""

    def __init__(self, tokenizer, morf_model, special_token_ids: dict, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.morf_model = morf_model
        self.max_seq_len = max_seq_len
        self.special = special_token_ids  # {"<|system|>": id, "<|user|>": id, ...}

    def _encode_text(self, text: str) -> list[int]:
        """Segment with Morfessor then tokenize. No BOS/EOS added."""
        segmented = segment_text(text, self.morf_model)
        return self.tokenizer.encode(segmented, add_bos=False, add_eos=False)

    def build_example(
        self,
        system_instruction: str,
        context_turns: list[dict],  # [{"speaker": "User", "text": "..."}, ...]
        user_input: str,
        assistant_output: str,
    ) -> SFTExample:
        """Build a single tokenized SFT example.

        Format:
            <bos> <|system|> {sys_tokens} <|end|>
            <|user|> {ctx_user_1} <|end|> <|assistant|> {ctx_asst_1} <|end|>
            ...
            <|user|> {current_input} <|end|>
            <|assistant|> {output} <|end|> <eos>

        Loss mask: 1 only on output tokens (after final <|assistant|>, up to <|end|><eos>).
        """
        ids = []
        mask = []

        def append_token(token_id, is_loss=False):
            ids.append(token_id)
            mask.append(1 if is_loss else 0)

        def append_tokens(token_ids, is_loss=False):
            ids.extend(token_ids)
            mask.extend([1 if is_loss else 0] * len(token_ids))

        # <bos>
        append_token(BOS_ID)

        # <|system|> {system_instruction} <|end|>
        append_token(self.special["<|system|>"])
        sys_ids = self._encode_text(system_instruction)
        append_tokens(sys_ids)
        append_token(self.special["<|end|>"])

        # Context turns (all masked — no loss on history)
        for turn in context_turns:
            speaker = turn["speaker"]
            text = turn["text"]
            if speaker == "User":
                append_token(self.special["<|user|>"])
                append_tokens(self._encode_text(text))
                append_token(self.special["<|end|>"])
            elif speaker == "Assistant":
                append_token(self.special["<|assistant|>"])
                append_tokens(self._encode_text(text))
                append_token(self.special["<|end|>"])

        # Current user input (masked)
        append_token(self.special["<|user|>"])
        append_tokens(self._encode_text(user_input))
        append_token(self.special["<|end|>"])

        # Assistant output (LOSS = 1)
        append_token(self.special["<|assistant|>"])
        output_ids = self._encode_text(assistant_output)
        append_tokens(output_ids, is_loss=True)
        append_token(self.special["<|end|>"], is_loss=True)

        # <eos>
        append_token(EOS_ID, is_loss=True)

        # Truncate if too long
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
            # Make sure we end cleanly — set last token to EOS
            ids[-1] = EOS_ID
            mask[-1] = 1

        return SFTExample(input_ids=ids, loss_mask=mask)


# ============================================================================
# Windowed Expansion
# ============================================================================
def expand_conversation(
    conversation: dict,
    builder: ChatTemplateBuilder,
    context_window: int = 4,
) -> list[SFTExample]:
    """Expand a multi-turn conversation into windowed SFT examples.

    For a conversation with N turn pairs, produces N training examples.
    Each example has up to `context_window` previous turns as context.

    Args:
        conversation: Raw conversation dict from translated.jsonl
        builder: ChatTemplateBuilder instance
        context_window: Max number of previous turns to include as context

    Returns:
        List of SFTExample instances
    """
    system_instruction = conversation["system_instruction"]
    turns = conversation["conversation"]

    # Pair turns into (User, Assistant) pairs
    pairs = []
    i = 0
    while i < len(turns) - 1:
        if turns[i]["speaker"] == "User" and turns[i + 1]["speaker"] == "Assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            # Skip malformed turns
            logger.warning(
                "Skipping malformed turn at index %d in %s: expected User/Assistant pair",
                i, conversation.get("sample_id", "unknown")
            )
            i += 1

    examples = []
    for pair_idx, (user_turn, asst_turn) in enumerate(pairs):
        # Build context: last `context_window` turns before this pair
        # Context window is in terms of individual turns (not pairs)
        # So context_window=4 means 2 User + 2 Assistant = 2 pairs
        context_pairs_count = context_window // 2
        start = max(0, pair_idx - context_pairs_count)
        context_turns = []
        for prev_pair_idx in range(start, pair_idx):
            context_turns.append(pairs[prev_pair_idx][0])  # User
            context_turns.append(pairs[prev_pair_idx][1])  # Assistant

        example = builder.build_example(
            system_instruction=system_instruction,
            context_turns=context_turns,
            user_input=user_turn["text"],
            assistant_output=asst_turn["text"],
        )
        examples.append(example)

    return examples


# ============================================================================
# Dataset Splitting
# ============================================================================
def split_conversations(
    conversations: list[dict],
    train_ratio: float = 0.96,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split at conversation level to avoid data leakage."""
    rng = random.Random(seed)
    indices = list(range(len(conversations)))
    rng.shuffle(indices)

    n = len(indices)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [conversations[i] for i in train_idx]
    val = [conversations[i] for i in val_idx]
    test = [conversations[i] for i in test_idx]

    return train, val, test


# ============================================================================
# Binary I/O
# ============================================================================
def save_split(
    examples: list[SFTExample],
    output_dir: Path,
    split_name: str,
) -> dict:
    """Save tokenized examples as flat binary files.

    Files:
        {split}.bin     — uint32 token IDs (all examples concatenated)
        {split}.mask.bin — uint8 loss masks (parallel to token IDs)
        {split}.offsets.json — byte offsets for each example

    Returns:
        Stats dict for this split.
    """
    all_ids = []
    all_masks = []
    offsets = []  # (start_idx, length) for each example
    total_loss_tokens = 0
    total_tokens = 0

    for ex in examples:
        start = total_tokens
        all_ids.extend(ex.input_ids)
        all_masks.extend(ex.loss_mask)
        total_tokens += ex.length
        total_loss_tokens += sum(ex.loss_mask)
        offsets.append({"start": start, "length": ex.length})

    # Save token IDs
    ids_arr = np.array(all_ids, dtype=np.uint32)
    ids_path = output_dir / f"{split_name}.bin"
    ids_arr.tofile(str(ids_path))

    # Save loss masks
    mask_arr = np.array(all_masks, dtype=np.uint8)
    mask_path = output_dir / f"{split_name}.mask.bin"
    mask_arr.tofile(str(mask_path))

    # Save offsets
    offsets_path = output_dir / f"{split_name}.offsets.json"
    with open(offsets_path, "w") as f:
        json.dump(offsets, f)

    stats = {
        "examples": len(examples),
        "total_tokens": total_tokens,
        "loss_tokens": total_loss_tokens,
        "loss_ratio": total_loss_tokens / total_tokens if total_tokens > 0 else 0,
        "avg_seq_len": total_tokens / len(examples) if examples else 0,
        "min_seq_len": min(ex.length for ex in examples) if examples else 0,
        "max_seq_len": max(ex.length for ex in examples) if examples else 0,
        "ids_file": str(ids_path),
        "mask_file": str(mask_path),
        "offsets_file": str(offsets_path),
    }
    return stats


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build SFT dataset for Pothana Base 300M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Path to translated.jsonl (full conversations)",
    )
    parser.add_argument(
        "--tokenizer", "-t", type=Path, required=True,
        help="Path to tokenizer directory or tokenizer.json",
    )
    parser.add_argument(
        "--morfessor", "-m", type=Path, required=True,
        help="Path to Morfessor binary model (.bin)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for binary files",
    )
    parser.add_argument(
        "--context-window", type=int, default=4,
        help="Number of previous turns to include as context (default: 4 = 2 pairs)",
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.96,
        help="Train split ratio (default: 0.96)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.02,
        help="Validation split ratio (default: 0.02)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing files",
    )
    args = parser.parse_args()

    # ---- Load conversations ----
    logger.info("Loading conversations from %s", args.input)
    conversations = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                skipped += 1
                continue

            meta = row.get("metadata", {})

            # Skip untranslated or errored rows
            if not meta.get("translated", True):
                skipped += 1
                continue
            if meta.get("translation_error", False):
                skipped += 1
                continue

            # Validate conversation structure
            conv = row.get("conversation", [])
            if not isinstance(conv, list) or len(conv) < 2:
                logger.warning("Skipping %s: conversation too short", row.get("sample_id", f"line_{line_num}"))
                skipped += 1
                continue

            # Check for empty text
            has_empty = any(not turn.get("text", "").strip() for turn in conv)
            if has_empty:
                logger.warning("Skipping %s: contains empty turn text", row.get("sample_id", f"line_{line_num}"))
                skipped += 1
                continue

            conversations.append(row)

    logger.info("Loaded %d conversations (%d skipped)", len(conversations), skipped)

    if not conversations:
        logger.error("No valid conversations found!")
        sys.exit(1)

    # ---- Load tokenizer and Morfessor ----
    logger.info("Loading tokenizer from %s", args.tokenizer)
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer

    tokenizer = MorfessorTokenizer(args.tokenizer)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)

    logger.info("Loading Morfessor model from %s", args.morfessor)
    morf_model = load_morfessor_model(args.morfessor)

    # ---- Assign special token IDs dynamically (after existing vocab) ----
    base_vocab_size = tokenizer.vocab_size
    special_token_ids = {}
    for i, name in enumerate(NEW_SPECIAL_TOKEN_NAMES):
        special_token_ids[name] = base_vocab_size + i
    new_vocab_size = base_vocab_size + len(NEW_SPECIAL_TOKEN_NAMES)

    logger.info("  New special tokens: %s", special_token_ids)
    logger.info("  New vocab size: %d (was %d)", new_vocab_size, base_vocab_size)

    # ---- Build template builder ----
    builder = ChatTemplateBuilder(
        tokenizer=tokenizer,
        morf_model=morf_model,
        special_token_ids=special_token_ids,
        max_seq_len=args.max_seq_len,
    )

    # ---- Split conversations ----
    logger.info("Splitting conversations (seed=%d): %.0f%% train / %.0f%% val / %.0f%% test",
                args.seed, args.train_ratio * 100, args.val_ratio * 100,
                (1 - args.train_ratio - args.val_ratio) * 100)

    train_convos, val_convos, test_convos = split_conversations(
        conversations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    logger.info("  Train: %d conversations", len(train_convos))
    logger.info("  Val:   %d conversations", len(val_convos))
    logger.info("  Test:  %d conversations", len(test_convos))

    # ---- Expand conversations into windowed examples ----
    from tqdm import tqdm

    def expand_split(convos, split_name):
        examples = []
        truncated = 0
        for conv in tqdm(convos, desc=f"Expanding {split_name}"):
            exs = expand_conversation(conv, builder, context_window=args.context_window)
            for ex in exs:
                if ex.length == args.max_seq_len:
                    truncated += 1
                examples.append(ex)
        if truncated:
            logger.info("  %s: %d examples truncated to max_seq_len=%d",
                        split_name, truncated, args.max_seq_len)
        return examples

    train_examples = expand_split(train_convos, "train")
    val_examples = expand_split(val_convos, "val")
    test_examples = expand_split(test_convos, "test")

    # ---- Print stats ----
    def print_stats(examples, name):
        if not examples:
            logger.info("  %s: 0 examples", name)
            return
        total_tokens = sum(ex.length for ex in examples)
        loss_tokens = sum(sum(ex.loss_mask) for ex in examples)
        lengths = [ex.length for ex in examples]
        logger.info("  %s: %d examples, %d tokens (%.1f%% loss), "
                     "seq_len: min=%d avg=%.0f max=%d",
                     name, len(examples), total_tokens,
                     100 * loss_tokens / total_tokens if total_tokens else 0,
                     min(lengths), sum(lengths) / len(lengths), max(lengths))

    logger.info("Dataset statistics:")
    print_stats(train_examples, "train")
    print_stats(val_examples, "val")
    print_stats(test_examples, "test")

    total_examples = len(train_examples) + len(val_examples) + len(test_examples)
    logger.info("  Total: %d examples from %d conversations", total_examples, len(conversations))

    if args.dry_run:
        logger.info("Dry run — not writing files.")
        return

    # ---- Save binary files ----
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to %s", args.output.resolve())

    train_stats = save_split(train_examples, args.output, "train")
    val_stats = save_split(val_examples, args.output, "val")
    test_stats = save_split(test_examples, args.output, "test")

    # ---- Save metadata ----
    meta = {
        "source": str(args.input),
        "tokenizer": str(args.tokenizer),
        "morfessor": str(args.morfessor),
        "context_window": args.context_window,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1 - args.train_ratio - args.val_ratio, 4),
        },
        "conversations": {
            "total": len(conversations),
            "skipped": skipped,
            "train": len(train_convos),
            "val": len(val_convos),
            "test": len(test_convos),
        },
        "special_tokens": {
            **special_token_ids,
            "<bos>": BOS_ID,
            "<eos>": EOS_ID,
        },
        "vocab_size_original": base_vocab_size,
        "vocab_size_with_special": new_vocab_size,
        "new_special_tokens": NEW_SPECIAL_TOKEN_NAMES,
        "splits": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
    }

    meta_path = args.output / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata to %s", meta_path)

    # ---- Verify ----
    logger.info("\n=== Verification ===")
    # Build reverse lookup for new special tokens
    id_to_special = {v: k for k, v in special_token_ids.items()}
    id_to_special[BOS_ID] = "<bos>"
    id_to_special[EOS_ID] = "<eos>"

    # Decode first training example to check formatting
    if train_examples:
        ex = train_examples[0]
        decoded_parts = []
        for tid in ex.input_ids[:100]:  # first 100 tokens
            special_name = id_to_special.get(tid)
            if special_name:
                decoded_parts.append(special_name)
            else:
                tok_str = tokenizer.id_to_token.get(tid, f"[UNK:{tid}]")
                decoded_parts.append(tok_str)
        logger.info("First example (first 100 tokens):")
        logger.info("  Tokens: %s", " ".join(decoded_parts))
        logger.info("  Mask:   %s", " ".join(str(m) for m in ex.loss_mask[:100]))
        logger.info("  Length: %d tokens, %d loss tokens",
                     ex.length, sum(ex.loss_mask))

    logger.info("\nDone! Files saved to %s", args.output.resolve())
    logger.info("Next step: use these with the SFT training script.")
    logger.info("Remember: embedding layer must be resized from %d to %d "
                 "to accommodate new special tokens: %s",
                 base_vocab_size, new_vocab_size, NEW_SPECIAL_TOKEN_NAMES)


if __name__ == "__main__":
    main()
