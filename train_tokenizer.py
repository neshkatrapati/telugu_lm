#!/usr/bin/env python3
"""
Telugu Morfessor + BPE Tokenizer Builder (v2)
==============================================
Builds a unified tokenizer that handles:
  - Telugu morphemes from Morfessor (with @@ continuation markers preserved)
  - Non-Telugu text (English, numbers, URLs) via BPE subword encoding
  - Character-level fallback for anything not covered

Key change from v1: @@ is part of the token, not stripped.
  - "విద్యార్థు@@" (ID=X) and "విద్యార్థు" (ID=Y) are SEPARATE vocab entries.
  - This lets the model learn word boundaries and decode reconstructs perfectly.

Pipeline:
  1. Scan segmented corpus — collect ALL tokens (with @@ preserved)
  2. Load BPE vocab (from train_bpe.py) — merge into unified token set
  3. Add character-level fallback (both ch and ch@@ variants)
  4. Build token-to-id / id-to-token mappings
  5. Save as JSON tokenizer (v2.0)

Usage:
    python train_tokenizer.py \\
        --segmented-corpus ./data/morfessor/segmented_corpus/sangraha/ \\
        --bpe-vocab ./data/morfessor/bpe/bpe_vocab.tsv \\
        --bpe-merges ./data/morfessor/bpe/bpe_merges.txt \\
        --output ./tokenizer

    python train_tokenizer.py \\
        --segmented-corpus ./data/morfessor/segmented_corpus/sangraha/ \\
        --output ./tokenizer
"""

import re
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import OrderedDict, Counter

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

TELUGU_CHAR_RE = re.compile(r"[\u0C00-\u0C7F]")
TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


# ---------------------------------------------------------------------------
# Step 1: Build vocabulary from segmented corpus
# ---------------------------------------------------------------------------
def build_vocab_from_corpus(corpus_path: Path, separator: str) -> list[tuple[str, int]]:
    """Scan segmented corpus files and count every unique token AS-IS.

    CRITICAL: Tokens are NOT stripped of @@. "విద్యార్థు@@" and "విద్యార్థు"
    are counted as separate entries. This is the v2 fix — the model can now
    learn word boundaries.

    Args:
        corpus_path: Path to a .seg.txt file or a directory containing them.
        separator: The morpheme boundary marker (e.g. '@@'). Used for logging only.

    Returns:
        List of (token, frequency) tuples sorted by frequency descending.
    """
    from tqdm import tqdm

    if corpus_path.is_file():
        seg_files = [corpus_path]
    else:
        seg_files = sorted(corpus_path.rglob("*.seg.txt"))

    if not seg_files:
        logger.error("No .seg.txt files found in %s", corpus_path)
        sys.exit(1)

    logger.info("Scanning %d segmented file(s) to build vocabulary...", len(seg_files))
    token_freq: Counter = Counter()
    total_tokens = 0

    for fpath in seg_files:
        logger.info("  Scanning %s", fpath.name)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=fpath.name, unit=" lines"):
                for token in line.split():
                    if token:
                        token_freq[token] += 1
                        total_tokens += 1

    # Count how many are continuation vs word-final
    continuation = sum(1 for t in token_freq if t.endswith(separator))
    word_final = len(token_freq) - continuation

    logger.info("Scanned %d total tokens", total_tokens)
    logger.info("Unique token types: %d (%d continuation, %d word-final)",
                len(token_freq), continuation, word_final)

    morphemes = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    return morphemes


# ---------------------------------------------------------------------------
# Step 2: Load BPE vocab and merges
# ---------------------------------------------------------------------------
def load_bpe_vocab(vocab_path: Path) -> dict[str, int]:
    """Load BPE vocabulary from TSV file (from train_bpe.py)."""
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                vocab[parts[0]] = int(parts[1])
    logger.info("Loaded %d BPE subwords from %s", len(vocab), vocab_path)
    return vocab


def load_bpe_merges(merges_path: Path) -> list[tuple[str, str]]:
    """Load BPE merge rules from file (from train_bpe.py)."""
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    logger.info("Loaded %d BPE merge rules from %s", len(merges), merges_path)
    return merges


# ---------------------------------------------------------------------------
# Step 3: Build unified tokenizer
# ---------------------------------------------------------------------------
def build_tokenizer(
    output_dir: Path,
    separator: str,
    segmented_corpus: Path = None,
    morfessor_dir: Path = None,
    vocab_size: int = 0,
    bpe_vocab_path: Path = None,
    bpe_merges_path: Path = None,
):
    """Build a unified tokenizer from Morfessor morphemes + BPE subwords.

    Vocabulary structure:
        [special tokens] + [morfessor morphemes with @@] + [BPE subwords] + [char fallbacks]

    Args:
        output_dir: Where to save tokenizer files.
        separator: Continuation marker (default: @@).
        segmented_corpus: Path to .seg.txt files (recommended, ensures zero UNK).
        morfessor_dir: Fallback — directory containing morpheme_vocab.tsv.
        vocab_size: Cap vocab at this size (0 = use all).
        bpe_vocab_path: Path to bpe_vocab.tsv from train_bpe.py.
        bpe_merges_path: Path to bpe_merges.txt from train_bpe.py.
    """

    # --- Collect Morfessor morphemes ---
    if segmented_corpus is not None:
        morphemes = build_vocab_from_corpus(segmented_corpus, separator)
    elif morfessor_dir is not None:
        vocab_path = morfessor_dir / "morpheme_vocab.tsv"
        if not vocab_path.exists():
            logger.error("morpheme_vocab.tsv not found at %s", vocab_path)
            logger.error("Run: python morfessor_segment.py --input ./data --train-only")
            logger.error("Or use --segmented-corpus to build vocab from segmented text files")
            sys.exit(1)

        logger.info("Reading morpheme vocabulary from %s", vocab_path)
        morphemes = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    morphemes.append((parts[0], int(parts[1])))
    else:
        logger.error("Must provide either --segmented-corpus or --morfessor-dir")
        sys.exit(1)

    logger.info("Morfessor morphemes: %d types", len(morphemes))

    # Sort by frequency
    morphemes.sort(key=lambda x: x[1], reverse=True)

    # Cap if requested
    if vocab_size > 0:
        max_morphemes = vocab_size - NUM_SPECIAL
        if len(morphemes) > max_morphemes:
            logger.info("Capping morphemes from %d to %d", len(morphemes), max_morphemes)
            morphemes = morphemes[:max_morphemes]

    # --- Build token-to-id mapping ---
    token_to_id = dict(SPECIAL_TOKENS)
    id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}
    next_id = NUM_SPECIAL

    # Add morfessor morphemes (these include both "x@@" and "x" variants)
    morfessor_count = 0
    for morph, freq in morphemes:
        if morph not in token_to_id:
            token_to_id[morph] = next_id
            id_to_token[next_id] = morph
            next_id += 1
            morfessor_count += 1

    logger.info("Added %d Morfessor morpheme tokens (IDs %d-%d)",
                morfessor_count, NUM_SPECIAL, next_id - 1)

    # --- Add BPE subwords (non-Telugu) ---
    bpe_merges = []
    bpe_count = 0
    if bpe_vocab_path and bpe_merges_path:
        bpe_vocab = load_bpe_vocab(bpe_vocab_path)
        bpe_merges = load_bpe_merges(bpe_merges_path)

        # Add BPE subwords that aren't already in the vocab
        # BPE produces bare subwords — we need both "sub" and "sub@@" variants
        for subword, freq in sorted(bpe_vocab.items(), key=lambda x: -x[1]):
            # Add the bare subword (word-final form)
            if subword not in token_to_id:
                token_to_id[subword] = next_id
                id_to_token[next_id] = subword
                next_id += 1
                bpe_count += 1
            # Add the continuation form (subword@@)
            cont_form = subword + separator
            if cont_form not in token_to_id:
                token_to_id[cont_form] = next_id
                id_to_token[next_id] = cont_form
                next_id += 1
                bpe_count += 1

        logger.info("Added %d BPE subword tokens", bpe_count)
    else:
        logger.info("No BPE vocab provided — non-Telugu text will use character fallback")

    # --- Add character-level fallback ---
    # For any token not covered by morphemes or BPE, we fall back to characters.
    # We add both "ch" (word-final) and "ch@@" (continuation) variants.
    char_ranges = []
    # Printable ASCII (32-126)
    char_ranges.extend(chr(c) for c in range(32, 127))
    # Telugu Unicode block (0C00-0C7F)
    char_ranges.extend(chr(c) for c in range(0x0C00, 0x0C80))
    # Common punctuation & symbols
    char_ranges.extend(list("\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u20ac\u20b9\u00b0\u00b1\u00d7\u00f7"))

    char_count = 0
    for ch in char_ranges:
        # Word-final form
        if ch not in token_to_id:
            token_to_id[ch] = next_id
            id_to_token[next_id] = ch
            next_id += 1
            char_count += 1
        # Continuation form (ch@@)
        cont_ch = ch + separator
        if cont_ch not in token_to_id:
            token_to_id[cont_ch] = next_id
            id_to_token[next_id] = cont_ch
            next_id += 1
            char_count += 1

    final_vocab_size = len(token_to_id)

    logger.info("Added %d character fallback tokens", char_count)
    logger.info("")
    logger.info("Tokenizer built (v2.0):")
    logger.info("  Vocab size:       %d", final_vocab_size)
    logger.info("  Special tokens:   %d", NUM_SPECIAL)
    logger.info("  Morfessor tokens: %d", morfessor_count)
    logger.info("  BPE tokens:       %d", bpe_count)
    logger.info("  Char fallbacks:   %d", char_count)

    # --- Save tokenizer ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save as JSON (human-readable)
    tokenizer_json = {
        "version": "2.0",
        "type": "morfessor_bpe_telugu",
        "vocab_size": final_vocab_size,
        "separator": separator,
        "special_tokens": dict(SPECIAL_TOKENS),
        "token_to_id": token_to_id,
        "bpe_merges": [[a, b] for a, b in bpe_merges],
    }
    json_path = output_dir / "tokenizer.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer JSON to %s", json_path)

    # 2. Save vocab list (one token per line)
    vocab_list_path = output_dir / "vocab.txt"
    with open(vocab_list_path, "w", encoding="utf-8") as f:
        for tid in range(final_vocab_size):
            f.write(f"{id_to_token[tid]}\n")
    logger.info("Saved vocab list to %s", vocab_list_path)

    # 3. Save token frequencies (for analysis)
    freq_path = output_dir / "token_frequencies.tsv"
    with open(freq_path, "w", encoding="utf-8") as f:
        f.write("token_id\ttoken\tfrequency\n")
        for name, tid in SPECIAL_TOKENS.items():
            f.write(f"{tid}\t{name}\t0\n")
        for i, (morph, freq) in enumerate(morphemes):
            tid = token_to_id.get(morph)
            if tid is not None:
                f.write(f"{tid}\t{morph}\t{freq}\n")
    logger.info("Saved token frequencies to %s", freq_path)

    return token_to_id, id_to_token, final_vocab_size


# ---------------------------------------------------------------------------
# BPE encode helper (used at encode time for non-Telugu tokens)
# ---------------------------------------------------------------------------
def bpe_encode_word(word: str, merges: list[tuple[str, str]], separator: str = "@@") -> list[str]:
    """Encode a word into BPE subwords using the learned merge table.

    Returns subwords with @@ on non-final pieces:
        "international" -> ["inter@@", "nation@@", "al"]
    """
    if not word:
        return []

    chars = list(word)

    for a, b in merges:
        merged = a + b
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars

    # Add @@ to all subwords except the last (word-final)
    result = []
    for i, subword in enumerate(chars):
        if i < len(chars) - 1:
            result.append(subword + separator)
        else:
            result.append(subword)

    return result


# ---------------------------------------------------------------------------
# Tokenizer class (for use by training script and inference)
# ---------------------------------------------------------------------------
class MorfessorTokenizer:
    """
    Unified tokenizer for Morfessor-segmented Telugu + BPE non-Telugu text.

    v2: @@ is part of the token string. "విద్యార్థు@@" and "విద్యార్థు" have
    different token IDs. This means decode is trivial: just join and replace
    "@@ " with "".

    Expects input text that has already been segmented (by morfessor_segment.py
    or inference.py's segment_text), with @@ continuation markers.

    Example:
        segmented = "విద్యార్థు@@ ల@@ కు went to school"
        ids = tokenizer.encode(segmented)
        text = tokenizer.decode(ids)
        # text == "విద్యార్థులకు went to school"
    """

    def __init__(self, tokenizer_path: str | Path):
        tokenizer_path = Path(tokenizer_path)

        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"

        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.version = data.get("version", "1.0")
        self.vocab_size = data["vocab_size"]
        self.separator = data["separator"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in
                           {v: k for k, v in data["token_to_id"].items()}.items()}
        # Rebuild id_to_token properly: iterate token_to_id
        self.id_to_token = {}
        for token, tid in self.token_to_id.items():
            self.id_to_token[tid] = token

        self.special_tokens = data["special_tokens"]

        self.pad_id = self.special_tokens["<pad>"]
        self.unk_id = self.special_tokens["<unk>"]
        self.bos_id = self.special_tokens["<bos>"]
        self.eos_id = self.special_tokens["<eos>"]

        # Load BPE merges if present
        self.bpe_merges = []
        if "bpe_merges" in data and data["bpe_merges"]:
            self.bpe_merges = [(a, b) for a, b in data["bpe_merges"]]
            logger.info("Loaded %d BPE merge rules from tokenizer", len(self.bpe_merges))

    def _is_telugu(self, token: str) -> bool:
        """Check if a token contains any Telugu characters."""
        return bool(TELUGU_CHAR_RE.search(token))

    def _encode_token_bpe(self, word: str) -> list[int]:
        """Encode a single non-Telugu word using BPE merges, with char fallback."""
        if self.bpe_merges:
            subwords = bpe_encode_word(word, self.bpe_merges, self.separator)
            ids = []
            for sw in subwords:
                tid = self.token_to_id.get(sw)
                if tid is not None:
                    ids.append(tid)
                else:
                    # BPE subword not in vocab — char fallback
                    # Strip @@ to check if it's a continuation piece
                    is_cont = sw.endswith(self.separator)
                    bare = sw[:-len(self.separator)] if is_cont else sw
                    for ci, ch in enumerate(bare):
                        is_last_char = (ci == len(bare) - 1) and not is_cont
                        if not is_last_char:
                            cid = self.token_to_id.get(ch + self.separator, self.unk_id)
                        else:
                            cid = self.token_to_id.get(ch, self.unk_id)
                        ids.append(cid)
            return ids
        else:
            # No BPE — pure character fallback
            return self._encode_token_chars(word)

    def _encode_token_chars(self, word: str) -> list[int]:
        """Encode a word character-by-character with @@ continuation."""
        ids = []
        for i, ch in enumerate(word):
            if i < len(word) - 1:
                # Continuation character
                cid = self.token_to_id.get(ch + self.separator)
                if cid is None:
                    cid = self.token_to_id.get(ch, self.unk_id)
                ids.append(cid)
            else:
                # Word-final character
                cid = self.token_to_id.get(ch, self.unk_id)
                ids.append(cid)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        """Encode segmented text to token IDs.

        The text should already be segmented by morfessor_segment.py with @@
        continuation markers. Each whitespace-separated token is looked up
        directly in the vocabulary (with @@ preserved).

        For non-Telugu tokens not in the vocab, BPE encoding is attempted,
        then character fallback.

        Args:
            text: Segmented text, e.g. "విద్యార్థు@@ ల@@ కు went to school"
            add_bos: Prepend <bos> token.
            add_eos: Append <eos> token.

        Returns:
            List of integer token IDs.
        """
        ids = []
        if add_bos:
            ids.append(self.bos_id)

        for token in text.split():
            if not token:
                continue

            # Direct lookup — this is the primary path
            # Token includes @@ if it's a continuation piece
            tid = self.token_to_id.get(token)
            if tid is not None:
                ids.append(tid)
                continue

            # Token not in vocab — need fallback
            # Strip @@ to get the bare form for BPE/char encoding
            is_continuation = token.endswith(self.separator)
            bare = token[:-len(self.separator)] if is_continuation else token

            if not self._is_telugu(bare):
                # Non-Telugu: try BPE encoding on the bare form
                sub_ids = self._encode_token_bpe(bare)
                if is_continuation and sub_ids:
                    # The last sub-token should be continuation, not word-final
                    # Replace the last ID with its @@ variant if possible
                    last_token_str = self.id_to_token.get(sub_ids[-1], "")
                    if not last_token_str.endswith(self.separator):
                        cont_tid = self.token_to_id.get(last_token_str + self.separator)
                        if cont_tid is not None:
                            sub_ids[-1] = cont_tid
                ids.extend(sub_ids)
            else:
                # Telugu token not in vocab — character fallback
                if is_continuation:
                    # Encode chars, last char gets @@ (since whole token is continuation)
                    for i, ch in enumerate(bare):
                        if i < len(bare) - 1:
                            cid = self.token_to_id.get(ch + self.separator, self.unk_id)
                        else:
                            # Last char of a continuation token — still gets @@
                            cid = self.token_to_id.get(ch + self.separator,
                                                       self.token_to_id.get(ch, self.unk_id))
                        ids.append(cid)
                else:
                    # Word-final: last char is bare
                    for i, ch in enumerate(bare):
                        if i < len(bare) - 1:
                            cid = self.token_to_id.get(ch + self.separator, self.unk_id)
                        else:
                            cid = self.token_to_id.get(ch, self.unk_id)
                        ids.append(cid)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        Reconstruction is trivial in v2: tokens with @@ are continuation
        pieces that join to the next token. Replace "@@ " with "" to merge.

        Args:
            ids: List of integer token IDs.

        Returns:
            Reconstructed text string.
        """
        tokens = []
        for tid in ids:
            token = self.id_to_token.get(tid, "<unk>")
            if token in ("<pad>", "<bos>", "<eos>"):
                continue
            tokens.append(token)

        # Join with spaces, then merge continuation pieces
        # "విద్యార్థు@@ ల@@ కు" -> "విద్యార్థులకు"
        text = " ".join(tokens)
        text = text.replace(self.separator + " ", "")
        return text

    def __len__(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# Test tokenization
# ---------------------------------------------------------------------------
def test_tokenizer(tokenizer_dir: Path, test_texts: list[str], separator: str):
    """Test the tokenizer on sample texts."""
    tokenizer = MorfessorTokenizer(tokenizer_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOKENIZER TEST (v%s)", tokenizer.version)
    logger.info("=" * 70)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)
    logger.info("  BPE merges: %d", len(tokenizer.bpe_merges))
    logger.info("")

    # If we have a morfessor model, we can segment the test texts
    morfessor_model_path = tokenizer_dir.parent / "data" / "morfessor" / "morfessor_telugu.bin"
    # Also check relative to tokenizer dir
    alt_path = Path("./data/morfessor/morfessor_telugu.bin")
    model = None
    for mpath in [morfessor_model_path, alt_path]:
        if mpath.exists():
            try:
                import morfessor
                io = morfessor.MorfessorIO()
                model = io.read_binary_model_file(str(mpath))
                logger.info("  (Using Morfessor model for test segmentation: %s)", mpath)
                break
            except ImportError:
                pass

    for text in test_texts:
        # Segment with Morfessor if available
        if model:
            seg_tokens = []
            for word in text.split():
                if TELUGU_WORD_RE.fullmatch(word):
                    segments = model.viterbi_segment(word)[0]
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            seg_tokens.append(seg + separator)
                        else:
                            seg_tokens.append(seg)
                else:
                    seg_tokens.append(word)
            segmented = " ".join(seg_tokens)
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

        # Verify round-trip
        if text == decoded.strip():
            logger.info("  Round-trip: PASS")
        else:
            logger.info("  Round-trip: MISMATCH")
            logger.info("    Expected: '%s'", text)
            logger.info("    Got:      '%s'", decoded.strip())
        logger.info("")

    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build Telugu tokenizer from Morfessor morphemes + BPE (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full build with BPE:
  %(prog)s --segmented-corpus ./data/morfessor/segmented_corpus/sangraha/ \\
           --bpe-vocab ./data/morfessor/bpe/bpe_vocab.tsv \\
           --bpe-merges ./data/morfessor/bpe/bpe_merges.txt

  # Without BPE (character fallback for non-Telugu):
  %(prog)s --segmented-corpus ./data/morfessor/segmented_corpus/sangraha/

  # From morpheme_vocab.tsv:
  %(prog)s --morfessor-dir ./data/morfessor

  # Test:
  %(prog)s --test "తెలుగు భాష చాలా అందమైనది"
        """,
    )

    parser.add_argument(
        "--segmented-corpus", type=str, default=None,
        help="Path to segmented corpus file or directory (.seg.txt). "
             "Builds vocab directly from the corpus (recommended).",
    )
    parser.add_argument(
        "--morfessor-dir", type=str, default="./data/morfessor",
        help="Directory containing morpheme_vocab.tsv (fallback if no --segmented-corpus).",
    )
    parser.add_argument(
        "--output", type=str, default="./tokenizer",
        help="Output directory for tokenizer files (default: ./tokenizer).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=0,
        help="Cap vocabulary size (0 = use all morphemes, default: 0).",
    )
    parser.add_argument(
        "--separator", type=str, default="@@",
        help="Continuation marker (default: @@).",
    )
    parser.add_argument(
        "--bpe-vocab", type=str, default=None,
        help="Path to BPE vocabulary TSV file (from train_bpe.py).",
    )
    parser.add_argument(
        "--bpe-merges", type=str, default=None,
        help="Path to BPE merge rules file (from train_bpe.py).",
    )
    parser.add_argument(
        "--test", type=str, nargs="*", default=None,
        help="Test sentences to tokenize.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    seg_corpus = Path(args.segmented_corpus) if args.segmented_corpus else None
    morfessor_dir = Path(args.morfessor_dir) if args.morfessor_dir else None
    bpe_vocab_path = Path(args.bpe_vocab) if args.bpe_vocab else None
    bpe_merges_path = Path(args.bpe_merges) if args.bpe_merges else None

    # Validate BPE args — need both or neither
    if (bpe_vocab_path is None) != (bpe_merges_path is None):
        logger.error("Must provide both --bpe-vocab and --bpe-merges, or neither")
        sys.exit(1)

    # Build tokenizer
    build_tokenizer(
        output_dir=output_dir,
        separator=args.separator,
        segmented_corpus=seg_corpus,
        morfessor_dir=morfessor_dir,
        vocab_size=args.vocab_size,
        bpe_vocab_path=bpe_vocab_path,
        bpe_merges_path=bpe_merges_path,
    )

    # Test
    test_texts = args.test or [
        "తెలుగు భాష చాలా అందమైనది",
        "విద్యార్థులకు మంచి విద్య అవసరం",
        "ప్రభుత్వం కొత్త పథకాన్ని ప్రారంభించింది",
        "భారతదేశంలో అనేక భాషలు మాట్లాడతారు",
    ]

    test_tokenizer(output_dir, test_texts, args.separator)

    logger.info("")
    logger.info("Tokenizer v2.0 ready at %s", output_dir.resolve())
    logger.info("  vocab.txt             — one token per line")
    logger.info("  tokenizer.json        — full tokenizer config + BPE merges")
    logger.info("  token_frequencies.tsv — token ID + frequency")


if __name__ == "__main__":
    main()
