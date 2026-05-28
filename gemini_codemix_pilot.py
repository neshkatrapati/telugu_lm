#!/usr/bin/env python3
"""
Stage A+ codemix data — Gemini pilot script.

Reads N chunks from train.bin, decodes to clean Telugu text, calls Gemini
(via toktee proxy) to get three formats per chunk, writes JSONL output.

Usage:
    python gemini_codemix_pilot.py \\
        --train-bin /workspace/telugu_lm/train-data-v2/train.bin \\
        --tokenizer-src /workspace/telugu_lm/tokenizer-new-v5 \\
        --out /workspace/telugu_lm/codemix-pilot.jsonl \\
        --n-chunks 1000 \\
        --chunk-size 512 \\
        --workers 8
"""
import argparse
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------- Token bookkeeping ----------

SPECIAL_TOKEN_STRS = {"<pad>", "<unk>", "<bos>", "<eos>",
                       "<search>", "</search>", "<retrieved>", "</retrieved>",
                       "<doc>", "</doc>", "<cite>", "<think>", "</think>"}


def load_inverse_vocab(tokenizer_json: Path) -> dict:
    data = json.load(open(tokenizer_json))
    return {v: k for k, v in data["token_to_id"].items()}


def decode_chunk_clean(ids, id_to_token) -> str:
    """Decode token IDs to clean readable Telugu text (no specials, no @@)."""
    parts = []
    for tid in ids:
        tok = id_to_token.get(int(tid))
        if tok is None or tok in SPECIAL_TOKEN_STRS:
            continue
        parts.append(tok)
    text = " ".join(parts)
    # @@ continuation prefix → join to previous
    text = text.replace(" @@", "").replace("@@", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Prompt ----------

PROMPT_TEMPLATE = """You are helping build training data for a Telugu language model that will be deployed in a mobile app. Indian users mix Telugu and English heavily in real typing — this data teaches the model to read and write that natural mix.

Given a Telugu passage, produce THREE rewritten versions:

1. "codemix_te_en"
   Natural code-mixing where TELUGU words stay in Telugu script and ENGLISH words appear in English script. This mirrors how educated Telugu speakers type on WhatsApp / social media — Telugu sentence structure preserved, but commonly English-loaned concepts (especially nouns: meeting, app, movie, review, brand names, technical terms) appear in English.

   Example:
     Input:  ఈ సినిమా చాలా బాగుంది, నేను సమీక్ష చదివాను
     Output: ఈ movie చాలా బాగుంది, నేను review చదివాను

2. "codemix_roman"
   The SAME natural code-mixing as above, but ALL TELUGU WORDS are now written in Roman/Latin script (phone-typed Tenglish style). English words stay in English script too. The result reads like a real WhatsApp message.

   Example (matching the above):
     Output: ee movie chala bagundi, nenu review chadivanu

3. "telugu_roman"
   The ORIGINAL Telugu content (no English substitution) rendered entirely in Roman script — natural phone-typed Tenglish. Same meaning as input, just different script.

   Example (matching the above):
     Output: ee sinima chala bagundi, nenu samiksha chadivanu

GUIDELINES for "natural" Telugu-English code-mixing (formats 1 and 2):

- Substitute English where Telugu speakers ACTUALLY do — nouns like "meeting", "app", "movie", "review", "ticket", "office", "WhatsApp", "phone", "school", "doctor", brand names, technical terms.
- Do NOT substitute culturally-Telugu words: amma/nanna/akka, food names, religious terms, common verbs/postpositions, pronouns.
- Substitute 15-40% of substitutable nouns — not every word. Real users mix selectively.
- Preserve Telugu sentence structure and verb forms (అయింది, చేశాను, etc.)
- Punctuation stays the same (periods, commas, question marks).

GUIDELINES for Roman Telugu (formats 2 and 3):

⚠ MOST IMPORTANT RULE — ABSOLUTELY NO CAPITAL LETTERS for Telugu sounds.
Telugu words must be ALL LOWERCASE. Capital letters appear ONLY for:
   (a) the first letter of a proper noun (Hyderabad, Annamayya, Pratap)
   (b) English words that are normally capitalized (WhatsApp, YouTube, NCERT)
   (c) explicit acronyms (CM, DGP, IIT)

Capital letters used to indicate long vowels (e.g., "nEnu", "lEdu", "rEvu",
"nApalLE", "annamaiyA") are FORBIDDEN. Real Telugu speakers do NOT type
capital E, O, A, I, U mid-word. This is academic ITRANS notation and is wrong
for our purpose. ALWAYS write the lowercase phonetic equivalent.

BAD                            GOOD
─────────────────────────────  ─────────────────────────────
nEnu repu                      nenu repu
ee teevra ardika lEdu          ee teevra ardika ledu
rEvuku chErina                 revuku cherina
vEstundi pravakta              vestundi pravaktha
samvatsaraalu                  samvatsaralu (or samvatsaraalu — both lowercase)
nEnpathylO                     nepathyalo
ChEsina yoda                   chesina yoda

Other Roman style rules:
- Use "ch" for చ (chala, chesa, chudu — NOT chAla, CHala).
- Use "th" for ధ where natural, "d" for ద (thana, dudu).
- Drop retroflex marks (.d → d, .t → t): write "manchi" not "ma.mci".
- Drop anusvara dots: "kavyam" not "kavyaM" or "kavya.m".
- Long vowels: usually just write the single letter (naku, leda, rendu).
  Doubled vowels (naaku, leedaa) are OK occasionally but should be the
  minority. NEVER use capital letters to mark long vowels.
- "umdi" or "undi" are both fine for ఉంది.

Read aloud — if it doesn't look like how a Telugu friend would WhatsApp on
their phone (all lowercase, casual, simple spelling), it's wrong.

EDGE CASES:
- If the input ALREADY contains English words: keep them as English in all three outputs; don't translate them back to Telugu.
- If the input is very short (< 10 words): still produce all three; if any format would be identical to input, that's fine.
- If the input contains code, URLs, numbers, or hashtags: leave them unchanged in all formats.
- Preserve approximate length — don't add or remove content.
- If the input is ARCHAIC / SANSKRIT / RELIGIOUS classical text and there are
  genuinely no natural English substitutions, "codemix_te_en" may be identical
  to the input — that's acceptable. But Roman outputs (formats 2 and 3) must
  still be properly Romanized in lowercase phone-typed style.

OUTPUT FORMAT — strict JSON, no markdown, no explanation:

{
  "codemix_te_en": "...",
  "codemix_roman": "...",
  "telugu_roman": "..."
}

NOW PROCESS THIS INPUT:

<<<INPUT>>>
%TELUGU_INPUT%
<<<END>>>"""


# ---------- Gemini call ----------

def make_client(api_key: str):
    """Return a configured Gemini client using the toktee proxy."""
    from google import genai
    from google.genai import types
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url="https://toktee.athlytesports.com",
            headers={"x-tokentee-task": "pothana-codemix-pilot"},
        ),
    )


BATCHED_PROMPT_HEAD = """You are helping build training data for a Telugu language model that will be deployed in a mobile app. Indian users mix Telugu and English heavily in real typing — this data teaches the model to read and write that natural mix.

For EACH Telugu passage below, produce THREE rewritten versions:

1. "codemix_te_en"
   Natural code-mixing where TELUGU words stay in Telugu script and ENGLISH words appear in English script. Common English-loaned concepts (meeting, app, movie, review, brand names, tech terms) appear in English. Substitute 15-40% of substitutable nouns. Preserve Telugu sentence structure and verb forms.

   Example:
     Input:  ఈ సినిమా చాలా బాగుంది, నేను సమీక్ష చదివాను
     Output: ఈ movie చాలా బాగుంది, నేను review చదివాను

2. "codemix_roman"
   The SAME code-mixing as above, but ALL TELUGU WORDS in Roman script. English words stay in English script. Lowercase, phone-typed Tenglish style.

   Example: ee movie chala bagundi, nenu review chadivanu

3. "telugu_roman"
   The ORIGINAL Telugu content (no English substitution) rendered entirely in Roman script — natural phone-typed Tenglish.

   Example: ee sinima chala bagundi, nenu samiksha chadivanu

⚠ ROMAN STYLE RULES (formats 2 and 3):
- ALL LOWERCASE for Telugu sounds. Capitals ONLY for proper nouns (Hyderabad, Annamayya) and English words (WhatsApp, NCERT).
- FORBIDDEN: capital E, O, A, I, U mid-word for long vowels. Write "ledu" not "lEdu", "revu" not "rEvu", "jivitamlo" not "jIvitamlO".
- Use "ch" for చ, drop retroflex dots, drop anusvara dots ("kavyam" not "kavyaM").

⚠ CODEMIX RULES (formats 1 and 2):
- Substitute English nouns where Telugu speakers naturally do.
- Do NOT substitute: family terms (amma/nanna), food, religious terms, pronouns, verb endings.
- Preserve punctuation, numbers, URLs unchanged.

EDGE: If a passage is archaic/Sanskrit/religious and has no natural English substitutions, "codemix_te_en" may equal the input — that's OK. Roman outputs still need proper lowercase Romanization.

OUTPUT FORMAT — strict JSON ARRAY. One object per input in the SAME ORDER as the inputs. No markdown, no commentary.

[
  {"codemix_te_en": "...", "codemix_roman": "...", "telugu_roman": "..."},
  ...
]

NOW PROCESS THESE INPUTS:

"""


def build_batched_prompt(texts: list[str]) -> str:
    p = BATCHED_PROMPT_HEAD
    for i, t in enumerate(texts, start=1):
        p += f"<<<INPUT {i}>>>\n{t}\n<<<END {i}>>>\n\n"
    return p


def parse_batched_response(text: str, expected_n: int) -> list[dict] | None:
    """Parse a JSON array of N entries, each having the three keys. Tolerant to markdown."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return None
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(arr, list) or len(arr) != expected_n:
        return None
    for obj in arr:
        if not isinstance(obj, dict):
            return None
        if not all(k in obj for k in ("codemix_te_en", "codemix_roman", "telugu_roman")):
            return None
    return arr


def call_gemini_batched(client, model: str, texts: list[str], max_retries: int = 3) -> list[dict] | str:
    """Call Gemini with batched prompt. Return list of dicts on success, or error string."""
    prompt = build_batched_prompt(texts)
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            parsed = parse_batched_response(response.text, len(texts))
            if parsed is not None:
                return parsed
            last_err = "parse_failed_or_count_mismatch"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return last_err or "unknown_error"


# ---------- Content filtering ----------

# Skip chunks dominated by archaic / Sanskrit / religious / fragmented content.
# These don't benefit from code-mixing (no modern English substitutions apply)
# and are noisy for training the routing/synthesis behaviors we care about.

_SANSKRIT_PUNCT = re.compile(r"[॥।]")  # devanagari/Sanskrit verse separators
_VISARGA_HEAVY = re.compile(r"ః")        # Telugu visarga — common in Sanskrit-loan
_ARCHAIC_WORDS_RE = re.compile(r"(ఇత్యథకాః|ఇతిచేతిచార్|ప్రత్యయః|స్తంభిత|విశిఖ|మృతా|మృత్యోః|స్వగ స్త్రీయ|శ్లోక|శ్లోకం|ఆయుర్దాయ)")
# Heuristic: ratio of single-char tokens (suggests heavy char-fallback fragmentation)

def is_archaic_or_low_quality(text: str) -> tuple[bool, str]:
    """Return (skip, reason)."""
    if not text or len(text) < 200:
        return True, "too_short"
    if len(text) > 4000:
        return True, "too_long"
    # Sanskrit verse markers
    n_sanskrit_punct = len(_SANSKRIT_PUNCT.findall(text))
    if n_sanskrit_punct >= 2:
        return True, "sanskrit_punctuation"
    # Heavy visarga density
    n_visarga = len(_VISARGA_HEAVY.findall(text))
    if n_visarga >= 8:
        return True, "visarga_heavy"
    # Archaic word markers
    if _ARCHAIC_WORDS_RE.search(text):
        return True, "archaic_words"
    # Fragmented (lots of single Telugu chars separated by spaces — sign of bad decode)
    # Count whitespace-separated tokens that are exactly 1 Telugu char
    toks = text.split()
    if len(toks) >= 30:
        single_te = sum(1 for t in toks if len(t) == 1 and 0x0c00 <= ord(t[0]) <= 0x0c7f)
        if single_te / len(toks) >= 0.30:
            return True, "fragmented_chars"
    return False, "ok"


# ---------- Worker pool ----------

def process_batch(args):
    """Process a batch of chunks via one Gemini call.

    args = (client, model, batch_items) where batch_items = [(chunk_id, text), ...]
    Returns: list of result records, one per input chunk.
    """
    client, model, batch_items = args
    if not batch_items:
        return []
    texts = [text for _, text in batch_items]
    result = call_gemini_batched(client, model, texts)
    if isinstance(result, str):
        # Error: mark each chunk in the batch as errored
        return [{"chunk_id": cid, "telugu_text": text, "_error": result}
                for cid, text in batch_items]
    # Success: zip results back to chunks
    out = []
    for (cid, text), entry in zip(batch_items, result):
        out.append({
            "chunk_id": cid,
            "telugu_text": text,
            "codemix_te_en": entry["codemix_te_en"],
            "codemix_roman": entry["codemix_roman"],
            "telugu_roman": entry["telugu_roman"],
        })
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--tokenizer-src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-chunks", type=int, default=1000)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=3,
                    help="Number of chunks per Gemini call (1=single, 3=recommended)")
    ap.add_argument("--model", type=str, default="gemini-2.0-flash")
    ap.add_argument("--api-key", type=str, default=os.environ.get("TOKTEE_TOKEN", ""))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-text-chars", type=int, default=2000,
                    help="Truncate Telugu text longer than this before sending to Gemini")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("Missing --api-key or TOKTEE_TOKEN env var")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---------- Sample chunks ----------
    print(f"[1/3] sampling {args.n_chunks} chunks from {args.train_bin}", flush=True)
    n_tokens = args.train_bin.stat().st_size // 4
    rng = np.random.RandomState(args.seed)
    max_start = n_tokens - args.chunk_size - 1
    stride = max_start // args.n_chunks
    base = np.arange(args.n_chunks, dtype=np.int64) * stride
    jitter = rng.randint(0, max(1, stride), size=args.n_chunks).astype(np.int64)
    offsets = (base + jitter).astype(np.int64)

    # ---------- Decode chunks ----------
    print(f"[2/3] decoding chunks", flush=True)
    data = np.memmap(str(args.train_bin), dtype=np.uint32, mode="r")
    id_to_token = load_inverse_vocab(args.tokenizer_src / "tokenizer.json")
    texts = []
    for cid, off in enumerate(offsets):
        ids = data[off : off + args.chunk_size].tolist()
        text = decode_chunk_clean(ids, id_to_token)
        if len(text) > args.max_text_chars:
            text = text[: args.max_text_chars].rsplit(" ", 1)[0]
        texts.append(text)
    print(f"      decoded {len(texts)} chunks  (avg len: {sum(len(t) for t in texts)//len(texts):,} chars)", flush=True)

    # ---------- Pre-filter archaic chunks ----------
    print(f"[3/4] pre-filtering archaic chunks", flush=True)
    valid_items = []
    pre_skip_records = []
    for cid, text in enumerate(texts):
        skip, reason = is_archaic_or_low_quality(text)
        if skip:
            pre_skip_records.append({"chunk_id": cid, "telugu_text": text, "_skip": reason})
        else:
            valid_items.append((cid, text))
    print(f"      {len(valid_items)} valid, {len(pre_skip_records)} pre-filtered", flush=True)

    # ---------- Build batches ----------
    batches = []
    for i in range(0, len(valid_items), args.batch_size):
        batches.append(valid_items[i : i + args.batch_size])
    print(f"      {len(batches)} batches of size up to {args.batch_size}", flush=True)

    # ---------- Call Gemini in parallel ----------
    print(f"[4/4] calling Gemini ({args.model}) with {args.workers} workers, batch={args.batch_size}", flush=True)
    client = make_client(args.api_key)

    work_items = [(client, args.model, batch) for batch in batches]

    results_lock = threading.Lock()
    n_done_chunks = 0
    n_ok = 0
    n_err = 0
    n_batches_done = 0
    total_chunks = len(valid_items)
    t0 = time.time()
    last_log = t0

    with open(args.out, "w", encoding="utf-8") as fout:
        # Write pre-skipped records first
        for rec in pre_skip_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_batch, item): item for item in work_items}
            for fut in as_completed(futures):
                try:
                    batch_records = fut.result()
                except Exception as e:
                    batch_records = [{"_error": f"future_failed: {e}"}]
                with results_lock:
                    for rec in batch_records:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_done_chunks += 1
                        if "_error" in rec or "_skip" in rec:
                            n_err += 1
                        else:
                            n_ok += 1
                    fout.flush()
                    n_batches_done += 1
                    if time.time() - last_log >= 15:
                        elapsed = time.time() - t0
                        rate_chunks = n_done_chunks / elapsed
                        eta_min = (total_chunks - n_done_chunks) / max(rate_chunks, 1e-9) / 60
                        print(f"      [{100*n_done_chunks/total_chunks:5.1f}%] "
                              f"chunks {n_done_chunks}/{total_chunks}  batches {n_batches_done}/{len(batches)}  "
                              f"ok={n_ok} err={n_err}  {rate_chunks:.1f} ch/s  ETA {eta_min:.1f}m",
                              flush=True)
                        last_log = time.time()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  total chunks: {args.n_chunks}")
    print(f"  pre-filtered: {len(pre_skip_records)}")
    print(f"  gemini ok:    {n_ok}")
    print(f"  gemini err:   {n_err}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
