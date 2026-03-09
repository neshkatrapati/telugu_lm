#!/usr/bin/env python3
"""
Telugu Morfessor + BPE Tokenizer Builder (v4 — reversed @@ prefix)
====================================================================
Builds a unified tokenizer that handles:
  - Telugu morphemes from Morfessor (bare roots + @@-prefixed suffixes)
  - Non-Telugu text (English, numbers, URLs) via BPE subword encoding
  - Character-level fallback for anything not covered

v4 changes from v3:
  - Eliminated ▁ separator token — word boundaries are implicit
  - Bare token = new word (space before when decoding)
  - @@-prefixed token = continuation (join to previous, no space)
  - Vocab has both bare and @@-prefixed forms for qualifying Telugu suffixes
  - Decode: scan tokens left-to-right, @@prefix → join, bare → space + token

Pipeline:
  1. Scan segmented corpus — collect ALL tokens (bare Telugu + @@-prefixed)
     Non-Telugu tokens are skipped (handled by BPE).
  2. Load BPE vocab (from train_bpe.py) — handles all non-Telugu text
  3. Add character-level fallback
  4. Build token-to-id / id-to-token mappings
  5. Save as JSON tokenizer (v4.0)

Usage:
    python train_tokenizer.py \\
        --segmented-corpus ./data/morfessor/sample.seg.txt \\
        --bpe-vocab ./data/morfessor/bpe/bpe_vocab.tsv \\
        --bpe-merges ./data/morfessor/bpe/bpe_merges.txt \\
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

def _count_chunk(args: tuple) -> tuple[Counter, Counter, int]:
    """Worker: count token frequencies in a chunk of lines.

    v4: Tokens are either bare (roots/standalone) or @@-prefixed (suffixes).
    Both types with Telugu chars are counted separately.
    Non-Telugu tokens (English, numbers, URLs) are skipped — handled by BPE.

    Returns (bare_telugu_freq, prefix_telugu_freq, total_tokens_seen).
    """
    lines = args[0]
    bare_freq: Counter = Counter()
    prefix_freq: Counter = Counter()
    total = 0
    for line in lines:
        for token in line.split():
            if not token:
                continue
            total += 1
            if token.startswith("@@"):
                # @@-prefixed suffix — check if Telugu
                base = token[2:]
                if _has_telugu(base):
                    prefix_freq[token] += 1
            else:
                # Bare token — count if Telugu
                if _has_telugu(token):
                    bare_freq[token] += 1
    return bare_freq, prefix_freq, total


def build_vocab_from_corpus(corpus_path: Path, num_workers: int = 0) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Scan segmented corpus files and count Telugu token frequencies.

    v4: Collects both bare Telugu morphemes and @@-prefixed suffixes separately.
    Non-Telugu text is skipped (handled by BPE).

    Parallelized: streams lines into chunks, dispatches to workers.

    Returns:
        (bare_morphemes, prefix_morphemes) — each a list of (token, freq) sorted by freq desc.
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

    logger.info("Scanning %d segmented file(s) for Telugu morphemes (%d workers)...",
                len(seg_files), num_workers)
    bare_freq: Counter = Counter()
    prefix_freq: Counter = Counter()
    total_tokens = 0

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
                        futures.append(executor.submit(_count_chunk, (current_chunk,)))
                        current_chunk = []
            if current_chunk:
                futures.append(executor.submit(_count_chunk, (current_chunk,)))

            pbar.set_description(f"{fpath.name} (merging {len(futures)} chunks)")

            for fut in as_completed(futures):
                bf, pf, total = fut.result()
                bare_freq += bf
                prefix_freq += pf
                total_tokens += total

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
                        if token.startswith("@@"):
                            base = token[2:]
                            if _has_telugu(base):
                                prefix_freq[token] += 1
                        else:
                            if _has_telugu(token):
                                bare_freq[token] += 1

    tel_tokens = sum(bare_freq.values()) + sum(prefix_freq.values())
    non_tel = total_tokens - tel_tokens
    logger.info("Total tokens scanned: %d", total_tokens)
    logger.info("Telugu bare tokens: %d (%.1f%%), unique: %d",
                sum(bare_freq.values()), 100 * sum(bare_freq.values()) / total_tokens if total_tokens else 0,
                len(bare_freq))
    logger.info("Telugu @@-prefixed tokens: %d (%.1f%%), unique: %d",
                sum(prefix_freq.values()), 100 * sum(prefix_freq.values()) / total_tokens if total_tokens else 0,
                len(prefix_freq))
    logger.info("Non-Telugu tokens skipped: %d (%.1f%%)", non_tel,
                100 * non_tel / total_tokens if total_tokens else 0)

    bare_morphemes = sorted(bare_freq.items(), key=lambda x: x[1], reverse=True)
    prefix_morphemes = sorted(prefix_freq.items(), key=lambda x: x[1], reverse=True)
    return bare_morphemes, prefix_morphemes


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
    segmented_corpus: Path = None,
    morfessor_dir: Path = None,
    vocab_size: int = 0,
    bpe_vocab_path: Path = None,
    bpe_merges_path: Path = None,
    num_workers: int = 0,
    min_freq: int = 2,
):
    """Build a unified tokenizer from Morfessor morphemes + BPE subwords.

    v4 Vocabulary structure:
        [special tokens] + [bare Telugu morphemes] + [@@-prefixed Telugu suffixes]
        + [BPE subwords] + [char fallbacks]

    Word boundary semantics:
      - Bare token → new word (space before during decode)
      - @@-prefixed token → continuation (join to previous, no space)

    Args:
        output_dir: Where to save tokenizer files.
        segmented_corpus: Path to .seg.txt files — scanned for Telugu morphemes.
        morfessor_dir: Fallback — directory containing morpheme_vocab.tsv.
        vocab_size: Cap vocab at this size (0 = use all).
        bpe_vocab_path: Path to bpe_vocab.tsv from train_bpe.py.
        bpe_merges_path: Path to bpe_merges.txt from train_bpe.py.
        min_freq: Minimum token frequency to include in vocab (default: 2).
    """

    # --- Collect Morfessor morphemes ---
    if segmented_corpus is not None:
        bare_morphemes, prefix_morphemes = build_vocab_from_corpus(segmented_corpus, num_workers)

        # Filter by minimum frequency
        if min_freq > 1:
            before_bare = len(bare_morphemes)
            bare_morphemes = [(tok, freq) for tok, freq in bare_morphemes if freq >= min_freq]
            before_pfx = len(prefix_morphemes)
            prefix_morphemes = [(tok, freq) for tok, freq in prefix_morphemes if freq >= min_freq]
            logger.info("Filtered by min_freq=%d: bare %d->%d, @@-prefixed %d->%d",
                        min_freq, before_bare, len(bare_morphemes),
                        before_pfx, len(prefix_morphemes))
    else:
        logger.error("Must provide --segmented-corpus")
        sys.exit(1)

    logger.info("Bare Telugu morphemes available: %d", len(bare_morphemes))
    logger.info("@@-prefixed Telugu suffixes available: %d", len(prefix_morphemes))

    # --- Load BPE if provided ---
    bpe_merges = []
    bpe_subwords = []  # (subword, freq) sorted by freq desc
    if bpe_vocab_path and bpe_merges_path:
        bpe_vocab = load_bpe_vocab(bpe_vocab_path)
        bpe_merges = load_bpe_merges(bpe_merges_path)
        bpe_subwords = sorted(bpe_vocab.items(), key=lambda x: -x[1])
    else:
        logger.info("No BPE vocab provided — non-Telugu text will use character fallback")

    # --- Character fallback pool ---
    char_pool = []
    # Printable ASCII (32-126)
    char_pool.extend(chr(c) for c in range(32, 127))
    # Telugu Unicode block (0C00-0C7F)
    char_pool.extend(chr(c) for c in range(0x0C00, 0x0C80))
    # Common punctuation & symbols
    char_pool.extend(list("\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u20ac\u20b9\u00b0\u00b1\u00d7\u00f7"))

    # --- Budget allocation when vocab_size is capped ---
    # Priority: special > chars > BPE > @@-prefix Telugu > bare Telugu (trim rarest bare)
    # Chars and BPE are essential for non-Telugu; @@-prefix are essential for suffix coverage.
    # Bare Telugu is the largest pool and most trimmable (rarest roots go to char fallback).
    n_chars_est = len(set(char_pool))  # upper bound (some may overlap with other tokens)
    n_bpe = len(bpe_subwords)
    n_prefix = len(prefix_morphemes)
    n_bare = len(bare_morphemes)

    if vocab_size > 0:
        # Reserved slots = special + chars + BPE + @@-prefix
        reserved = NUM_SPECIAL + n_chars_est + n_bpe + n_prefix
        bare_budget = vocab_size - reserved
        if bare_budget < n_bare:
            if bare_budget < 0:
                logger.warning("vocab_size=%d is too small! Need at least %d for non-bare tokens.",
                               vocab_size, reserved)
                bare_budget = max(1000, n_bare // 2)  # fallback
            logger.info("Capping bare Telugu from %d to %d (vocab_size=%d)",
                        n_bare, bare_budget, vocab_size)
            bare_morphemes = bare_morphemes[:bare_budget]
        else:
            logger.info("vocab_size=%d — all %d bare Telugu fit (budget: %d)",
                        vocab_size, n_bare, bare_budget)

    # --- Build token-to-id mapping ---
    token_to_id = dict(SPECIAL_TOKENS)
    id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}
    next_id = NUM_SPECIAL

    # Add bare Telugu morphemes (roots + standalone words)
    bare_count = 0
    for morph, freq in bare_morphemes:
        if morph not in token_to_id:
            token_to_id[morph] = next_id
            id_to_token[next_id] = morph
            next_id += 1
            bare_count += 1
    bare_end_id = next_id - 1

    logger.info("Added %d bare Telugu morpheme tokens (IDs %d-%d)",
                bare_count, NUM_SPECIAL, bare_end_id)

    # Add @@-prefixed Telugu suffixes
    prefix_count = 0
    for morph, freq in prefix_morphemes:
        if morph not in token_to_id:
            token_to_id[morph] = next_id
            id_to_token[next_id] = morph
            next_id += 1
            prefix_count += 1
    prefix_end_id = next_id - 1

    logger.info("Added %d @@-prefixed Telugu suffix tokens (IDs %d-%d)",
                prefix_count, bare_end_id + 1, prefix_end_id)

    # Add BPE subwords (non-Telugu)
    bpe_count = 0
    for subword, freq in bpe_subwords:
        if subword not in token_to_id:
            token_to_id[subword] = next_id
            id_to_token[next_id] = subword
            next_id += 1
            bpe_count += 1

    if bpe_count:
        logger.info("Added %d BPE subword tokens", bpe_count)

    # Add character-level fallback
    char_count = 0
    for ch in char_pool:
        if ch not in token_to_id:
            token_to_id[ch] = next_id
            id_to_token[next_id] = ch
            next_id += 1
            char_count += 1

    final_vocab_size = len(token_to_id)

    logger.info("Added %d character fallback tokens", char_count)
    logger.info("")
    logger.info("Tokenizer built (v4.0 — reversed @@ prefix):")
    logger.info("  Vocab size:          %d", final_vocab_size)
    logger.info("  Special tokens:      %d", NUM_SPECIAL)
    logger.info("  Bare Telugu tokens:  %d", bare_count)
    logger.info("  @@-prefix tokens:    %d", prefix_count)
    logger.info("  BPE tokens:          %d", bpe_count)
    logger.info("  Char fallbacks:      %d", char_count)

    # --- Save tokenizer ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save as JSON
    tokenizer_json = {
        "version": "4.0",
        "type": "morfessor_bpe_telugu_v4",
        "vocab_size": final_vocab_size,
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
        f.write("token_id\ttoken\tfrequency\tsource\n")
        for name, tid in SPECIAL_TOKENS.items():
            f.write(f"{tid}\t{name}\t0\tspecial\n")
        for morph, freq in bare_morphemes:
            tid = token_to_id.get(morph)
            if tid is not None:
                f.write(f"{tid}\t{morph}\t{freq}\tbare_telugu\n")
        for morph, freq in prefix_morphemes:
            tid = token_to_id.get(morph)
            if tid is not None:
                f.write(f"{tid}\t{morph}\t{freq}\tprefix_telugu\n")
    logger.info("Saved token frequencies to %s", freq_path)

    return token_to_id, id_to_token, final_vocab_size


# ---------------------------------------------------------------------------
# BPE encode helper (used at encode time for non-Telugu tokens)
# ---------------------------------------------------------------------------

_bpe_merge_ranks: dict[tuple[str, str], int] | None = None


def _get_merge_ranks(merges: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Build a priority dict from merge list (lower rank = merge first)."""
    global _bpe_merge_ranks
    if _bpe_merge_ranks is not None and len(_bpe_merge_ranks) == len(merges):
        return _bpe_merge_ranks
    _bpe_merge_ranks = {pair: i for i, pair in enumerate(merges)}
    return _bpe_merge_ranks


def bpe_encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a word into BPE subwords using the learned merge table.

    Uses priority-based pair merging: at each step, finds the highest-priority
    (lowest rank) pair present in the current symbols and merges it.

    v4: Returns bare subwords. Word boundaries handled by @@ prefix scheme.
        "international" -> ["inter", "nation", "al"]
    """
    if not word:
        return []

    ranks = _get_merge_ranks(merges)
    symbols = list(word)

    while len(symbols) > 1:
        best_pair = None
        best_rank = len(merges)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            r = ranks.get(pair)
            if r is not None and r < best_rank:
                best_rank = r
                best_pair = pair

        if best_pair is None:
            break

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

    return symbols


# ---------------------------------------------------------------------------
# Tokenizer class (for use by training script and inference)
# ---------------------------------------------------------------------------
class MorfessorTokenizer:
    """
    Unified tokenizer for Morfessor-segmented Telugu + BPE non-Telugu text.

    v4: Reversed @@ prefix scheme for word boundaries.
    - Bare token = new word (space before when decoding)
    - @@-prefixed token = continuation (join to previous, no space)
    - No ▁ separator token

    Expects input text that has already been segmented by morfessor_segment.py
    (v4 format with @@ prefix on suffixes).

    Example:
        segmented = "విద్యార్థు @@ల @@కు went to school"
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
        self.token_to_id = data["token_to_id"]
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

        # Cache for BPE encode results
        self._bpe_cache: dict[str, list[int]] = {}

    def _is_telugu(self, token: str) -> bool:
        """Check if a token contains any Telugu characters."""
        return bool(TELUGU_CHAR_RE.search(token))

    def _encode_token_bpe(self, word: str) -> list[int]:
        """Encode a single non-Telugu word using BPE merges, with char fallback.

        v4: BPE produces bare subwords. Results are cached.
        """
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return cached

        if self.bpe_merges:
            subwords = bpe_encode_word(word, self.bpe_merges)
            ids = []
            for sw in subwords:
                tid = self.token_to_id.get(sw)
                if tid is not None:
                    ids.append(tid)
                else:
                    for ch in sw:
                        cid = self.token_to_id.get(ch, self.unk_id)
                        ids.append(cid)
            self._bpe_cache[word] = ids
            return ids
        else:
            ids = self._encode_token_chars(word)
            self._bpe_cache[word] = ids
            return ids

    def _encode_token_chars(self, word: str) -> list[int]:
        """Encode a word character-by-character."""
        ids = []
        for ch in word:
            cid = self.token_to_id.get(ch, self.unk_id)
            ids.append(cid)
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        """Encode segmented text to token IDs.

        v4: Input text has bare roots and @@-prefixed suffixes.
        Example: "విద్యార్థు @@ల @@కు went to school"

        Each whitespace-separated token is looked up directly in the vocab.
        For unknown tokens, BPE or char fallback is used.

        Args:
            text: Segmented text in v4 format.
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

            # Direct lookup — primary path
            # Handles bare morphemes, @@-prefixed suffixes, BPE subwords, chars
            tid = self.token_to_id.get(token)
            if tid is not None:
                ids.append(tid)
                continue

            # Token not in vocab — need fallback
            if token.startswith("@@"):
                # @@-prefixed token not in vocab — try char fallback on the base
                base = token[2:]
                if self._is_telugu(base):
                    ids.extend(self._encode_token_chars(base))
                else:
                    ids.extend(self._encode_token_bpe(base))
            elif not self._is_telugu(token):
                # Non-Telugu: try BPE encoding
                ids.extend(self._encode_token_bpe(token))
            else:
                # Telugu token not in vocab — character fallback
                ids.extend(self._encode_token_chars(token))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def encode_lines_to_array(self, lines: list[str], add_eos: bool = True) -> tuple:
        """Batch-encode multiple lines into a flat uint32 array + stats.

        v4: Tokens are bare or @@-prefixed. Direct lookup for most tokens.
        Optimized for data preparation.

        Args:
            lines: List of segmented text lines (v4 format).
            add_eos: Append <eos> after each line.

        Returns:
            (np.uint32 array of all IDs, total_token_count, unk_count)
        """
        _get = self.token_to_id.get
        _unk = self.unk_id
        _eos = self.eos_id
        _is_tel = self._is_telugu
        _bpe = self._encode_token_bpe

        all_ids = []
        total = 0
        unk_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for token in line.split():
                # Fast path: direct vocab lookup (handles ~95%+ of tokens)
                tid = _get(token)
                if tid is not None:
                    all_ids.append(tid)
                    total += 1
                    if tid == _unk:
                        unk_count += 1
                    continue

                # Slow path: token not in vocab
                total += 1

                if token.startswith("@@"):
                    # @@-prefixed OOV — char fallback on base
                    base = token[2:]
                    for ch in base:
                        cid = _get(ch, _unk)
                        all_ids.append(cid)
                        if cid == _unk:
                            unk_count += 1
                elif not _is_tel(token):
                    # Non-Telugu → BPE (cached)
                    sub_ids = _bpe(token)
                    all_ids.extend(sub_ids)
                    unk_count += sum(1 for i in sub_ids if i == _unk)
                else:
                    # Telugu unknown → char fallback
                    for ch in token:
                        cid = _get(ch, _unk)
                        all_ids.append(cid)
                        if cid == _unk:
                            unk_count += 1

            if add_eos:
                all_ids.append(_eos)
                total += 1

        return np.array(all_ids, dtype=np.uint32), total, unk_count

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.

        v4: Reversed @@ prefix scheme.
          - @@-prefixed token → join directly to previous (no space)
          - Bare token → new word (space before)
          - Strip leading/trailing whitespace

        Args:
            ids: List of integer token IDs.

        Returns:
            Reconstructed text string.
        """
        parts = []
        for tid in ids:
            token = self.id_to_token.get(tid, "")
            if token in ("<pad>", "<bos>", "<eos>", "<unk>"):
                continue
            if token.startswith("@@"):
                # Continuation — join to previous (no space)
                parts.append(token[2:])
            else:
                # New word — space before
                parts.append(" ")
                parts.append(token)

        return "".join(parts).strip()

    def __len__(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# Test tokenization
# ---------------------------------------------------------------------------
def test_tokenizer(tokenizer_dir: Path, test_texts: list[str]):
    """Test the tokenizer on sample texts."""
    tokenizer = MorfessorTokenizer(tokenizer_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOKENIZER TEST (v%s)", tokenizer.version)
    logger.info("=" * 70)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)
    logger.info("  BPE merges: %d", len(tokenizer.bpe_merges))
    logger.info("")

    # Try loading Morfessor model for test segmentation
    model = None
    suffix_set = None
    for mpath in [Path("./data/morfessor/morfessor_telugu.bin"),
                  tokenizer_dir.parent / "data" / "morfessor" / "morfessor_telugu.bin"]:
        if mpath.exists():
            try:
                import morfessor
                io = morfessor.MorfessorIO()
                model = io.read_binary_model_file(str(mpath))
                logger.info("  (Using Morfessor model for test segmentation: %s)", mpath)
                # Try to load suffix_set
                for spath in [Path("./data/morfessor/suffix_set.json"),
                              mpath.parent / "suffix_set.json"]:
                    if spath.exists():
                        import json as _json
                        with open(spath, "r", encoding="utf-8") as f:
                            suffix_data = _json.load(f)
                        suffix_set = set(suffix_data["morphemes"])
                        logger.info("  (Loaded suffix set: %d morphemes)", len(suffix_set))
                        break
                break
            except ImportError:
                pass

    for text in test_texts:
        # Segment with Morfessor if available — v4 format
        if model:
            from morfessor_segment import (
                split_script_boundaries, TELUGU_WORD_RE as _TEL_RE,
                _segment_telugu_word, _format_word_v4,
            )
            seg_tokens = []
            cache = {}
            for word in text.split():
                parts = split_script_boundaries(word)
                for part in parts:
                    if _TEL_RE.fullmatch(part):
                        morphemes = _segment_telugu_word(cache, model, part)
                        seg_tokens.extend(_format_word_v4(morphemes, suffix_set))
                    else:
                        seg_tokens.append(part)
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
            logger.info("  Round-trip: PASS ✓")
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
        description="Build Telugu tokenizer from Morfessor morphemes + BPE (v4 — reversed @@ prefix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full build with BPE:
  %(prog)s --segmented-corpus ./data/morfessor/sample.seg.txt \\
           --bpe-vocab ./data/morfessor/bpe/bpe_vocab.tsv \\
           --bpe-merges ./data/morfessor/bpe/bpe_merges.txt

  # Without BPE (character fallback for non-Telugu):
  %(prog)s --segmented-corpus ./data/morfessor/sample.seg.txt

  # Test:
  %(prog)s --test "తెలుగు భాష చాలా అందమైనది"
        """,
    )

    parser.add_argument(
        "--segmented-corpus", type=str, default=None,
        help="Path to segmented corpus file or directory (.seg.txt). "
             "Builds vocab directly from the corpus.",
    )
    parser.add_argument(
        "--output", type=str, default="./tokenizer",
        help="Output directory for tokenizer files (default: ./tokenizer).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=0,
        help="Cap vocabulary size (0 = use all, default: 0).",
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
        help="Number of parallel workers for corpus scan (default: auto).",
    )
    parser.add_argument(
        "--min-freq", type=int, default=2,
        help="Minimum token frequency to include in vocab (default: 2).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    seg_corpus = Path(args.segmented_corpus) if args.segmented_corpus else None
    bpe_vocab_path = Path(args.bpe_vocab) if args.bpe_vocab else None
    bpe_merges_path = Path(args.bpe_merges) if args.bpe_merges else None

    # Validate BPE args — need both or neither
    if (bpe_vocab_path is None) != (bpe_merges_path is None):
        logger.error("Must provide both --bpe-vocab and --bpe-merges, or neither")
        sys.exit(1)

    # Build tokenizer
    build_tokenizer(
        output_dir=output_dir,
        segmented_corpus=seg_corpus,
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

    test_tokenizer(output_dir, test_texts)

    logger.info("")
    logger.info("Tokenizer v4.0 (reversed @@ prefix) ready at %s", output_dir.resolve())
    logger.info("  vocab.txt             — one token per line")
    logger.info("  tokenizer.json        — full tokenizer config + BPE merges")
    logger.info("  token_frequencies.tsv — token ID + frequency + source")


if __name__ == "__main__":
    main()
