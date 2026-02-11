#!/usr/bin/env python3
"""
Telugu Morfessor-based Tokenizer Builder
=========================================
Builds a tokenizer from Morfessor segmentation output.

Instead of training BPE from scratch, we use the morpheme vocabulary
produced by Morfessor as our subword units. This gives us a morphologically
aware tokenizer for Telugu.

Pipeline:
  1. Read morpheme_vocab.tsv (from morfessor_segment.py --vocab-stats)
  2. Rank by frequency, keep top N morphemes
  3. Add special tokens (<pad>, <eos>, <bos>, <unk>)
  4. Build token-to-id and id-to-token mappings
  5. Save as a simple JSON tokenizer + binary vocab

Requirements:
    pip install pyarrow tqdm

Usage:
    python train_tokenizer.py                                    # Use ALL morphemes
    python train_tokenizer.py --vocab-size 16384                 # Cap at 16K vocab
    python train_tokenizer.py --morfessor-dir ./data/morfessor   # Custom path
    python train_tokenizer.py --test "తెలుగు భాష చాలా అందమైనది"  # Test tokenization
"""

import os
import sys
import json
import argparse
import logging
import struct
from pathlib import Path
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = OrderedDict([
    ("<pad>", 0),
    ("<unk>", 1),
    ("<bos>", 2),
    ("<eos>", 3),
])

NUM_SPECIAL = len(SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Build tokenizer from morpheme vocab
# ---------------------------------------------------------------------------
def build_tokenizer(
    morfessor_dir: Path,
    vocab_size: int,
    output_dir: Path,
    separator: str,
):
    """Build tokenizer from Morfessor morpheme vocabulary.

    If vocab_size is 0, use ALL morphemes from the Morfessor output
    (i.e. vocab = special tokens + every morpheme type).
    If vocab_size > 0, cap the vocabulary at that size.
    """

    vocab_path = morfessor_dir / "morpheme_vocab.tsv"
    if not vocab_path.exists():
        logger.error("morpheme_vocab.tsv not found at %s", vocab_path)
        logger.error("Run: python morfessor_segment.py --input ./data --train-only")
        sys.exit(1)

    # Read morpheme vocab
    logger.info("Reading morpheme vocabulary from %s", vocab_path)
    morphemes = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                morph, freq = parts[0], int(parts[1])
                morphemes.append((morph, freq))

    logger.info("Read %d morphemes from vocabulary", len(morphemes))

    # Sort by frequency (should already be sorted, but ensure)
    morphemes.sort(key=lambda x: x[1], reverse=True)

    # Determine actual vocab size
    if vocab_size <= 0:
        # Use ALL morphemes — vocab = special tokens + all morphemes
        max_morphemes = len(morphemes)
        vocab_size = max_morphemes + NUM_SPECIAL
        logger.info("Using all %d morphemes (vocab_size = %d)", max_morphemes, vocab_size)
    else:
        max_morphemes = vocab_size - NUM_SPECIAL
        if len(morphemes) < max_morphemes:
            logger.info(
                "Morpheme vocab (%d) smaller than requested vocab size (%d). Using all morphemes.",
                len(morphemes), vocab_size,
            )
            max_morphemes = len(morphemes)
            vocab_size = max_morphemes + NUM_SPECIAL

    selected_morphemes = morphemes[:max_morphemes]
    dropped = len(morphemes) - max_morphemes

    # Build mappings
    token_to_id = dict(SPECIAL_TOKENS)
    id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}

    for i, (morph, freq) in enumerate(selected_morphemes):
        tid = i + NUM_SPECIAL
        token_to_id[morph] = tid
        id_to_token[tid] = morph

    logger.info("Tokenizer built:")
    logger.info("  Vocab size:       %d", vocab_size)
    logger.info("  Special tokens:   %d", NUM_SPECIAL)
    logger.info("  Morpheme tokens:  %d", max_morphemes)
    logger.info("  Dropped (rare):   %d", dropped)

    if selected_morphemes:
        min_freq = selected_morphemes[-1][1]
        max_freq = selected_morphemes[0][1]
        logger.info("  Freq range:       %d - %d", min_freq, max_freq)

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save as JSON (human-readable, for inspection)
    tokenizer_json = {
        "version": "1.0",
        "type": "morfessor_telugu",
        "vocab_size": vocab_size,
        "separator": separator,
        "special_tokens": dict(SPECIAL_TOKENS),
        "token_to_id": token_to_id,
    }
    json_path = output_dir / "tokenizer.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer JSON to %s", json_path)

    # 2. Save vocab list (one token per line, for quick lookup)
    vocab_list_path = output_dir / "vocab.txt"
    with open(vocab_list_path, "w", encoding="utf-8") as f:
        for tid in range(vocab_size):
            f.write(f"{id_to_token[tid]}\n")
    logger.info("Saved vocab list to %s", vocab_list_path)

    # 3. Save token frequencies (for analysis)
    freq_path = output_dir / "token_frequencies.tsv"
    with open(freq_path, "w", encoding="utf-8") as f:
        f.write("token_id\ttoken\tfrequency\n")
        for name, tid in SPECIAL_TOKENS.items():
            f.write(f"{tid}\t{name}\t0\n")
        for i, (morph, freq) in enumerate(selected_morphemes):
            f.write(f"{i + NUM_SPECIAL}\t{morph}\t{freq}\n")
    logger.info("Saved token frequencies to %s", freq_path)

    return token_to_id, id_to_token, vocab_size


# ---------------------------------------------------------------------------
# Tokenizer class (for use by training script)
# ---------------------------------------------------------------------------
class MorfessorTokenizer:
    """
    Simple tokenizer that maps Morfessor-segmented text to token IDs.

    Expects input text that has already been segmented by Morfessor,
    with morpheme boundaries marked by the separator (default: @@).

    Example:
        segmented = "విద్యార్థు@@ ల@@ కు went to school"
        ids = tokenizer.encode(segmented)
    """

    def __init__(self, tokenizer_path: str | Path):
        tokenizer_path = Path(tokenizer_path)

        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.separator = data["separator"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.special_tokens = data["special_tokens"]

        self.pad_id = self.special_tokens["<pad>"]
        self.unk_id = self.special_tokens["<unk>"]
        self.bos_id = self.special_tokens["<bos>"]
        self.eos_id = self.special_tokens["<eos>"]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        """
        Encode segmented text to token IDs.

        The text should already be Morfessor-segmented with @@ markers.
        Tokens ending with @@ are continuation tokens (like BPE).
        """
        ids = []
        if add_bos:
            ids.append(self.bos_id)

        # Split on whitespace — each token is either a morpheme or a
        # morpheme with @@ suffix (continuation marker)
        for token in text.split():
            # Strip separator for lookup but keep the original token as-is
            # since the vocab was built from the segmented output
            clean = token.rstrip(self.separator)
            # Try exact match first (with separator), then without
            tid = self.token_to_id.get(token)
            if tid is None:
                tid = self.token_to_id.get(clean)
            if tid is None:
                tid = self.unk_id
            ids.append(tid)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in ids:
            token = self.id_to_token.get(tid, "<unk>")
            if token in ("<pad>", "<bos>", "<eos>"):
                continue
            tokens.append(token)

        # Reconstruct: join tokens, then remove separator to merge morphemes
        text = " ".join(tokens)
        # "విద్యార్థు@@ ల@@ కు" -> "విద్యార్థులకు"
        text = text.replace(self.separator + " ", "")
        return text

    def __len__(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# Test tokenization
# ---------------------------------------------------------------------------
def test_tokenizer(tokenizer_dir: Path, test_texts: list[str], separator: str):
    """Test the tokenizer on sample texts."""
    import re

    tokenizer = MorfessorTokenizer(tokenizer_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOKENIZER TEST")
    logger.info("=" * 70)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)
    logger.info("")

    # If we have a morfessor model, we can segment the test texts
    morfessor_model_path = tokenizer_dir.parent / "morfessor" / "morfessor_telugu.bin"
    model = None
    if morfessor_model_path.exists():
        try:
            import morfessor
            io = morfessor.MorfessorIO()
            model = io.read_binary_model_file(str(morfessor_model_path))
            logger.info("  (Using Morfessor model for test segmentation)")
        except ImportError:
            pass

    TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")

    for text in test_texts:
        # Segment with Morfessor if available
        if model:
            tokens = text.split()
            segmented_tokens = []
            for token in tokens:
                if TELUGU_WORD_RE.fullmatch(token):
                    segments = model.viterbi_segment(token)[0]
                    if len(segments) > 1:
                        for i, seg in enumerate(segments):
                            if i < len(segments) - 1:
                                segmented_tokens.append(seg + separator)
                            else:
                                segmented_tokens.append(seg)
                    else:
                        segmented_tokens.append(token)
                else:
                    segmented_tokens.append(token)
            segmented = " ".join(segmented_tokens)
        else:
            segmented = text  # assume already segmented

        ids = tokenizer.encode(segmented)
        decoded = tokenizer.decode(ids)
        unk_count = sum(1 for i in ids if i == tokenizer.unk_id)

        logger.info("  Input:     %s", text)
        logger.info("  Segmented: %s", segmented)
        logger.info("  IDs:       %s", ids[:20])
        if len(ids) > 20:
            logger.info("             ... (%d total)", len(ids))
        logger.info("  Decoded:   %s", decoded)
        logger.info("  Tokens: %d, UNKs: %d", len(ids), unk_count)
        logger.info("")

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build Telugu tokenizer from Morfessor morpheme vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                     # Use ALL morphemes from Morfessor
  %(prog)s --vocab-size 16384                  # Cap at 16K vocab
  %(prog)s --test "తెలుగు భాష చాలా అందమైనది"  # Test tokenization
        """,
    )

    parser.add_argument(
        "--morfessor-dir",
        type=str,
        default="./data/morfessor",
        help="Directory containing morfessor output (morpheme_vocab.tsv) (default: ./data/morfessor)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tokenizer",
        help="Output directory for tokenizer files (default: ./tokenizer)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=0,
        help="Target vocabulary size including special tokens. 0 = use ALL morphemes (default: 0)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="@@",
        help="Morpheme boundary separator used in segmented corpus (default: @@)",
    )
    parser.add_argument(
        "--test",
        type=str,
        nargs="*",
        default=None,
        help="Test sentences to tokenize (space-separated words)",
    )

    args = parser.parse_args()

    morfessor_dir = Path(args.morfessor_dir)
    output_dir = Path(args.output)

    # Build tokenizer
    token_to_id, id_to_token, vocab_size = build_tokenizer(
        morfessor_dir, args.vocab_size, output_dir, args.separator,
    )

    # Test
    test_texts = args.test or [
        "తెలుగు భాష చాలా అందమైనది",
        "విద్యార్థులకు మంచి విద్య అవసరం",
        "ప్రభుత్వం కొత్త పథకాన్ని ప్రారంభించింది",
        "భారతదేశంలో అనేక భాషలు మాట్లాడతారు",
    ]

    test_tokenizer(output_dir, test_texts, args.separator)

    logger.info("Tokenizer ready at %s", output_dir.resolve())
    logger.info("  vocab.txt          — one token per line")
    logger.info("  tokenizer.json     — full tokenizer config")
    logger.info("  token_frequencies.tsv — token ID + frequency")


if __name__ == "__main__":
    main()
