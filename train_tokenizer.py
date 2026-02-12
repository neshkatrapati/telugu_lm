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
  1. Scan segmented corpus — collect only TELUGU tokens (with @@ preserved)
     Non-Telugu tokens (English, numbers, URLs) are skipped here.
  2. Load BPE vocab (from train_bpe.py) — handles all non-Telugu text
  3. Add character-level fallback (both ch and ch@@ variants)
  4. Build token-to-id / id-to-token mappings
  5. Save as JSON tokenizer (v2.0)

Expected vocab size: ~33K Telugu morphemes + ~8K BPE subwords + ~500 chars ≈ ~42K

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

import numpy as np
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

# Pre-compiled set of Telugu codepoints for fast membership test (no regex per token)
_TELUGU_CP_RANGE = range(0x0C00, 0x0C80)


def _has_telugu(s: str) -> bool:
    """Fast Telugu detection — checks codepoints directly, no regex."""
    for ch in s:
        if ord(ch) in _TELUGU_CP_RANGE:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 1: Build vocabulary from segmented corpus (parallelized)
# ---------------------------------------------------------------------------

def _count_chunk(args: tuple) -> tuple[Counter, int, int]:
    """Worker: count Telugu-containing token frequencies in a chunk of lines.

    Only tokens containing at least one Telugu character are counted.
    Non-Telugu tokens (English, numbers, URLs) are skipped — they are
    handled entirely by BPE subwords.

    Returns (telugu_token_freq, total_tokens_seen, telugu_tokens_counted).
    """
    lines, separator, sep_len = args
    freq: Counter = Counter()
    total = 0
    telugu_count = 0
    for line in lines:
        for token in line.split():
            if not token:
                continue
            total += 1
            # Strip @@ suffix to check for Telugu characters
            if token.endswith(separator):
                base = token[:-sep_len]
            else:
                base = token
            if _has_telugu(base):
                freq[token] += 1  # Keep @@ as part of the key
                telugu_count += 1
    return freq, total, telugu_count


def build_vocab_from_corpus(corpus_path: Path, separator: str, num_workers: int = 0) -> list[tuple[str, int]]:
    """Scan segmented corpus files and count TELUGU-CONTAINING tokens only.

    CRITICAL v2 changes:
      1. Tokens are NOT stripped of @@. "విద్యార్థు@@" and "విద్యార్థు"
         are counted as separate entries, so the model can learn word boundaries.
      2. Only tokens containing at least one Telugu character are counted.
         Non-Telugu text (English, numbers, URLs) is handled entirely by BPE.
         This prevents vocab explosion from millions of unique non-Telugu surface forms.

    Parallelized: streams lines into chunks, dispatches to workers.

    Args:
        corpus_path: Path to a .seg.txt file or a directory containing them.
        separator: The morpheme boundary marker (e.g. '@@').
        num_workers: Number of parallel workers (0 = auto).

    Returns:
        List of (token, frequency) tuples sorted by frequency descending.
    """
    from tqdm import tqdm
    from multiprocessing import cpu_count

    if corpus_path.is_file():
        seg_files = [corpus_path]
    else:
        seg_files = sorted(corpus_path.rglob("*.seg.txt"))

    if not seg_files:
        logger.error("No .seg.txt files found in %s", corpus_path)
        sys.exit(1)

    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    CHUNK_SIZE = 50_000  # lines per chunk
    sep_len = len(separator)

    logger.info("Scanning %d segmented file(s) for Telugu morphemes (%d workers)...",
                len(seg_files), num_workers)
    logger.info("  (Non-Telugu tokens skipped — handled by BPE)")
    token_freq: Counter = Counter()
    total_tokens = 0
    telugu_tokens = 0

    for fpath in seg_files:
        logger.info("  Scanning %s", fpath.name)

        if num_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            futures = []
            current_chunk = []
            line_count = 0

            pbar = tqdm(desc=fpath.name, unit=" lines")
            executor = ProcessPoolExecutor(max_workers=num_workers)

            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    current_chunk.append(line)
                    line_count += 1
                    pbar.update(1)
                    if len(current_chunk) >= CHUNK_SIZE:
                        futures.append(executor.submit(_count_chunk, (current_chunk, separator, sep_len)))
                        current_chunk = []
            if current_chunk:
                futures.append(executor.submit(_count_chunk, (current_chunk, separator, sep_len)))

            pbar.set_description(f"{fpath.name} (merging {len(futures)} chunks)")

            for fut in as_completed(futures):
                freq, total, tel_count = fut.result()
                token_freq += freq
                total_tokens += total
                telugu_tokens += tel_count

            executor.shutdown(wait=False)
            pbar.close()
            logger.info("    %d lines, %d chunks", line_count, len(futures))

        else:
            # Single-threaded with progress bar
            with open(fpath, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=fpath.name, unit=" lines"):
                    for token in line.split():
                        if not token:
                            continue
                        total_tokens += 1
                        # Strip @@ suffix to check for Telugu characters
                        if token.endswith(separator):
                            base = token[:-sep_len]
                        else:
                            base = token
                        if _has_telugu(base):
                            token_freq[token] += 1  # Keep @@ as part of the key
                            telugu_tokens += 1

    # Count how many are continuation vs word-final
    continuation = sum(1 for t in token_freq if t.endswith(separator))
    word_final = len(token_freq) - continuation

    logger.info("Total tokens scanned: %d", total_tokens)
    logger.info("Telugu tokens: %d (%.1f%%)", telugu_tokens,
                100 * telugu_tokens / total_tokens if total_tokens else 0)
    logger.info("Non-Telugu tokens skipped: %d (%.1f%%)", total_tokens - telugu_tokens,
                100 * (total_tokens - telugu_tokens) / total_tokens if total_tokens else 0)
    logger.info("Unique Telugu morpheme types: %d (%d continuation, %d word-final)",
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
    num_workers: int = 0,
    min_freq: int = 2,
):
    """Build a unified tokenizer from Morfessor morphemes + BPE subwords.

    Vocabulary structure:
        [special tokens] + [Telugu morphemes with @@] + [BPE subwords] + [char fallbacks]

    The corpus scan only collects Telugu-containing tokens (~33K morphemes).
    Non-Telugu text is handled entirely by BPE subwords (~8K).
    This gives a predictable vocab size of ~42K instead of millions.

    Args:
        output_dir: Where to save tokenizer files.
        separator: Continuation marker (default: @@).
        segmented_corpus: Path to .seg.txt files — scanned for Telugu morphemes only.
        morfessor_dir: Fallback — directory containing morpheme_vocab.tsv.
        vocab_size: Cap vocab at this size (0 = use all).
        bpe_vocab_path: Path to bpe_vocab.tsv from train_bpe.py.
        bpe_merges_path: Path to bpe_merges.txt from train_bpe.py.
        min_freq: Minimum token frequency to include in vocab (default: 2).
    """

    # --- Collect Morfessor morphemes ---
    if segmented_corpus is not None:
        morphemes = build_vocab_from_corpus(segmented_corpus, separator, num_workers)

        # Filter by minimum frequency — removes rare junk (URLs, garbage, hapax legomena)
        if min_freq > 1:
            before = len(morphemes)
            morphemes = [(tok, freq) for tok, freq in morphemes if freq >= min_freq]
            dropped = before - len(morphemes)
            logger.info("Filtered by min_freq=%d: %d -> %d types (%d rare tokens dropped)",
                        min_freq, before, len(morphemes), dropped)
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

    logger.info("Telugu morphemes to add: %d types", len(morphemes))

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

# Module-level cache for the merge priority dict (built once, reused)
_bpe_merge_ranks: dict[tuple[str, str], int] | None = None


def _get_merge_ranks(merges: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Build a priority dict from merge list (lower rank = merge first)."""
    global _bpe_merge_ranks
    if _bpe_merge_ranks is not None and len(_bpe_merge_ranks) == len(merges):
        return _bpe_merge_ranks
    _bpe_merge_ranks = {pair: i for i, pair in enumerate(merges)}
    return _bpe_merge_ranks


def bpe_encode_word(word: str, merges: list[tuple[str, str]], separator: str = "@@") -> list[str]:
    """Encode a word into BPE subwords using the learned merge table.

    Uses priority-based pair merging: at each step, finds the highest-priority
    (lowest rank) pair present in the current symbols and merges it. This is
    O(word_length² × log) instead of O(num_merges × word_length).

    Returns subwords with @@ on non-final pieces:
        "international" -> ["inter@@", "nation@@", "al"]
    """
    if not word:
        return []

    ranks = _get_merge_ranks(merges)
    symbols = list(word)

    while len(symbols) > 1:
        # Find the pair with the lowest merge rank in current symbols
        best_pair = None
        best_rank = len(merges)  # sentinel: worse than any real rank
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            r = ranks.get(pair)
            if r is not None and r < best_rank:
                best_rank = r
                best_pair = pair

        if best_pair is None:
            break  # no more merges possible

        # Merge all occurrences of best_pair
        a, b = best_pair
        merged = a + b
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    # Add @@ to all subwords except the last (word-final)
    result = []
    for i, subword in enumerate(symbols):
        if i < len(symbols) - 1:
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

        # Cache for BPE encode results — same word always produces same subwords
        # Avoids re-running 8K merge rules for repeated tokens (huge speedup)
        self._bpe_cache: dict[str, list[int]] = {}

    def _is_telugu(self, token: str) -> bool:
        """Check if a token contains any Telugu characters."""
        return bool(TELUGU_CHAR_RE.search(token))

    def _encode_token_bpe(self, word: str) -> list[int]:
        """Encode a single non-Telugu word using BPE merges, with char fallback.

        Results are cached — the same word always produces the same IDs,
        so we avoid re-running 8K merge rules for repeated tokens.
        """
        # Check cache first
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached

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
            self._bpe_cache[word] = ids
            return ids
        else:
            # No BPE — pure character fallback
            ids = self._encode_token_chars(word)
            self._bpe_cache[word] = ids
            return ids

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

    def encode_lines_to_array(self, lines: list[str], add_eos: bool = True) -> tuple:
        """Batch-encode multiple lines into a flat uint32 array + stats.

        Optimized for data preparation — avoids per-line Python overhead.
        Uses local variable references for hot-path speedup.

        Args:
            lines: List of segmented text lines.
            add_eos: Append <eos> after each line.

        Returns:
            (np.uint32 array of all IDs, total_token_count, unk_count)
        """
        # Local refs for hot-path (avoids repeated attribute lookups)
        _get = self.token_to_id.get
        _unk = self.unk_id
        _eos = self.eos_id
        _sep = self.separator
        _sep_len = len(_sep)
        _is_tel = self._is_telugu
        _bpe = self._encode_token_bpe
        _id2tok = self.id_to_token

        all_ids = []
        total = 0
        unk_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for token in line.split():
                # Fast path: direct vocab lookup (handles ~85%+ of tokens)
                tid = _get(token)
                if tid is not None:
                    all_ids.append(tid)
                    total += 1
                    if tid == _unk:
                        unk_count += 1
                    continue

                # Slow path: token not in vocab
                total += 1
                is_cont = token.endswith(_sep)
                bare = token[:-_sep_len] if is_cont else token

                if not _is_tel(bare):
                    # Non-Telugu → BPE (cached)
                    sub_ids = _bpe(bare)
                    if is_cont and sub_ids:
                        last_str = _id2tok.get(sub_ids[-1], "")
                        if not last_str.endswith(_sep):
                            ct = _get(last_str + _sep)
                            if ct is not None:
                                sub_ids[-1] = ct
                    all_ids.extend(sub_ids)
                    unk_count += sum(1 for i in sub_ids if i == _unk)
                else:
                    # Telugu unknown → char fallback
                    n = len(bare)
                    if is_cont:
                        for i, ch in enumerate(bare):
                            if i < n - 1:
                                all_ids.append(_get(ch + _sep, _unk))
                            else:
                                all_ids.append(_get(ch + _sep, _get(ch, _unk)))
                    else:
                        for i, ch in enumerate(bare):
                            if i < n - 1:
                                all_ids.append(_get(ch + _sep, _unk))
                            else:
                                all_ids.append(_get(ch, _unk))

            if add_eos:
                all_ids.append(_eos)
                total += 1

        return np.array(all_ids, dtype=np.uint32), total, unk_count

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
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers for corpus scan (default: auto = cpu_count - 1).",
    )
    parser.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum token frequency to include in vocab. Filters rare junk (default: 2).",
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
        num_workers=args.workers,
        min_freq=args.min_freq,
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
