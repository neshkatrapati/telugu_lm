# Training the Telugu LM for Retrieval-Augmented Generation

## Goal

Turn a fresh ~200–225M Telugu base model (trained separately, see prerequisites below) into a **retrieval-augmented assistant** that:

1. **Decides when it needs to look something up** (vs. answer from parametric memory)
2. **Emits a search query** as a tool call (web search, Telugu Wikipedia, custom corpora)
3. **Synthesizes a grounded answer in Telugu** from retrieved passages
4. **Cites every factual claim** to a specific retrieved chunk
5. **Refuses or abstains** when retrieved context is insufficient — never hallucinates

This is the dominant inference-time lever for small models. The literature is unambiguous: RETRO showed 25× parameter efficiency from retrieval, kNN-LM showed a 100M-token-trained model + 3B-token datastore beats training on all 3B tokens, and Self-RAG / RAFT / RA-DIT have demonstrated that **retrieval behavior is trainable as a behavior on top of a base LM** — not something that needs to be baked into pretraining.

---

## Prerequisite: Fresh Base Model

This plan assumes a clean Telugu base model trained explicitly for downstream retrieval FT, not the existing `pothana-engrams` checkpoint. Engrams are off the table (they were demonstrably ignored by the gate per `engram/out_hidden_ratio < 1%` — see `STATE.md`).

The base must include the following features so the retrieval FT pipeline doesn't need to retrofit them:

- **block_size = 4096** (long enough for query + multi-chunk retrieved context + answer)
- `rope_theta = 500000` (extended for 4K context)
- **Retrieval special tokens pre-allocated** in the tokenizer: `<search>`, `</search>`, `<retrieved>`, `</retrieved>`, `<doc>`, `</doc>`, `<cite>`, `<think>`, `</think>`. Trained alongside everything else during base pretraining (unused signal, but embeddings get a proper init via gradients on adjacent tokens).
- **~10% English data** mixed into pretraining (~150–300M tokens — Wikipedia + FineWeb-Edu sample) for cross-lingual capability. Telugu Wikipedia is too small alone; English Wikipedia is the realistic factual datastore.
- **Untied embeddings**, **z-loss**, **QK-norm** for stability at long context.
- Arch: h=768, layers=24 (with weight sharing → effective depth 48), GQA 16/4, SwiGLU, RMSNorm. Total ~225M params.
- Tokenizer: `morfessor_bpe_telugu_v4` (`tokenizer-new/`), vocab=47822, extended with the ~10 retrieval tokens above.
- Trained ~3–5 epochs over the ~3.4B-token mixed corpus (90% Telugu + 10% English).

This base model is **not** trained on retrieval-augmented sequences — it's just a strong Telugu LM that already has the right tokenizer/architecture/context-length for FT to graft retrieval behavior onto. See `STATE.md` and project memory for the rationale.

> **Alternative path** (parked, not pursued in v1): bake retrieval awareness into base pretraining itself, with ~30% of training sequences formatted as `<retrieved>[related chunk]</retrieved>[target]` using e5+FAISS neighbors over the corpus. This would absorb what is now Stage A below into the base run. See appendix.

---

## Non-Goals (Honest Scoping for 200M Scale)

A 200M model is small. Most production tool-using models are 7B+. We will **not** target:

- **Long agentic chains** (5+ search calls, plan-execute-replan). The model lacks the working memory.
- **Open-ended browsing** (clicking links, scrolling, multi-page synthesis). That's WebGPT/Operator territory and needs a much larger reasoner.
- **Tool selection across many tools.** We support 1–3 retrieval tools, not 50 APIs.
- **Code execution as a tool.** Out of scope.

What we *can* realistically achieve at 200M:

- **Single-hop retrieval** (one search → synthesize) — reliable
- **Two-hop retrieval** (one follow-up search after reading first result) — best-effort
- **Faithful synthesis with citations** — primary differentiator vs. base LM
- **Calibrated abstention** ("I don't know, the retrieved sources don't cover this")

If you want full agentic behavior, the 200M is the wrong vehicle and we'd need to revisit scale. Everything below is scoped to what 200M can actually deliver.

---

## Why Finetune (Not Bake Retrieval Into Base Pretraining)

| Approach | Pro | Con | Verdict |
|---|---|---|---|
| **Joint retriever+LM pretraining (Atlas / RETRO)** | Cleanest integration; model sees retrieval format from token 0; jointly tunes retriever | Needs ~T-scale tokens of retrieval-paired data to pay off; we have 3.4B; engineering complexity (joint retriever+LM training) is high | ❌ |
| **Retrieval-aware base pretraining (ICLM-style)** | Model sees `<retrieved>[chunk]</retrieved>[target]` from token 0; absorbs Stage A; same compute as plain pretraining | Adds data-prep complexity at the most expensive moment; harder to debug if base goes wrong; can't compare against plain-LM ablation easily | Parked (see appendix) |
| **Plain base + retrieval FT (chosen)** | Decouples base-LM correctness from retrieval correctness; standard literature pattern (Self-RAG, RAFT, RA-DIT); easier to iterate on retrieval pipeline without re-pretraining | Requires a brief Stage A continued-pretrain to teach the retrieval format before SFT | ✅ |
| **Pure SFT (skip Stage A entirely)** | Cheapest | Base has never seen `<retrieved>...</retrieved>` format; SFT alone teaches the *format* but not the *prior* that context-grounded answers are good. Empirically worse per RAFT/RA-DIT. | ❌ |

**The base model's Telugu LM ability is an asset.** Retrieval behavior is a thin layer on top of language modeling — query generation, reading comprehension over retrieved passages, citation. None of these require re-learning Telugu, so we keep base and retrieval-FT clearly separated.

---

## Pipeline Overview

```
[Stage 0: Fresh Telugu Base — ~225M, h=768, block=4096, retrieval tokens pre-allocated]
       │   (see Prerequisite section above)
       ▼
┌──────────────────────────────────────────────────────┐
│ Stage A: Retrieval-Aware Continued Pretraining       │ ~1–2B tokens, light
│ "Acclimate to <retrieved>...[target] format"         │ 1–2 days compute
│ Optional but recommended (~0.5 NLL on grounded eval) │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ Stage B: Tool-Use SFT  (FULL FINETUNE)               │ 100–500K examples
│ "Teach the model to emit <search> and synthesize"    │ 2–5 days compute
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ Stage C: Faithfulness DPO                            │ 30–100K pairs
│ "Prefer grounded over hallucinated answers"          │ 1–3 days compute
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ Stage D: Inference-Time Verifier                     │ Lightweight head
│ "Reject samples that aren't grounded in context"     │ Runs at inference
└──────────────────────────────────────────────────────┘
```

Each stage hardens against hallucination from a different angle: A acclimates the prior, B teaches the format, C calibrates preferences, D enforces at decode time.

**All retrieval training is full finetuning** (no LoRA / no adapters). At 200M scale, full FT is cheap, and constraining the model's parametric capacity is the last thing we want when teaching it a new behavior.

---

## Stage A: Retrieval-Aware Continued Pretraining (light)

### Why this stage exists

If we jump straight to SFT with `<retrieved>...</retrieved>` traces, the base model has never seen this format despite having the special tokens in its vocabulary. The token IDs exist but the *statistical regularity* that text inside `<retrieved>` tags is informative for what follows is not yet learned. SFT alone can teach the *surface format* but not the *prior* that "answer should be derivable from context". This is the gap RAFT (Zhang et al., 2024) and RA-DIT (Lin et al., 2023) both flag.

Continued pretraining on retrieval-augmented sequences fixes this. The model learns:
- Text inside `<retrieved>...</retrieved>` is semantically related to what follows
- Generation should be consistent with prepended context
- Distractor chunks can be in the retrieved block; attend selectively

This stage is **shorter and cheaper than in the original plan**, because (a) special tokens are already in the vocab from Stage 0, and (b) context length is already 4096 — no architecture changes needed.

### Data construction

Use the 3.4B-token mixed corpus (90% Telugu + 10% English from Stage 0), but **reformat** ~80% of it into retrieval-augmented sequences for this stage:

**Format 1 — Document continuation with related context (In-Context Pretraining, Shi et al. 2023):**
```
<retrieved>
<doc>[Related document chunk A from Telugu Wikipedia or news, retrieved by dense similarity]</doc>
<doc>[Related document chunk B]</doc>
</retrieved>
[Original document — the model predicts this normally]
```

**Format 2 — Chunked self-retrieval:**
```
<retrieved>
<doc>[Earlier chunk of THIS document]</doc>
<doc>[Distant chunk of THIS document, simulating long-context retrieval]</doc>
</retrieved>
[Current chunk — predict this]
```

**Format 3 — Distractor-aware (RAFT-style, ~30% of stage data):**
```
<retrieved>
<doc>[Relevant chunk]</doc>
<doc>[Irrelevant distractor chunk]</doc>
<doc>[Irrelevant distractor chunk]</doc>
</retrieved>
[Continuation that depends only on the relevant chunk]
```

The distractor mix is crucial. It teaches the model to **ignore noise** in retrieved context — a primary failure mode of small RAG models.

### Loss

Standard causal LM loss on every token (including inside `<retrieved>` blocks). No masking. The retrieval prior emerges from causal attention from target tokens backward to retrieved tokens — the gradient signal makes the model use that information. Free LM-style training; the retrieval semantics come from the *data distribution*, not the loss.

### Compute budget

- ~1–2B tokens of retrieval-formatted continued pretraining (vs. ~5–10B in the original plan — base model is already aware of long context + special tokens, so the format-learning is faster)
- 1 epoch over the reformatted ~2B-token subset
- **LR: 3e-5** (low, to avoid catastrophic forgetting of Telugu LM)
- Warmup: 500 steps (short; the LR is already low)
- ~1–2 days on B200

### Retriever for Stage A data construction

Use **multilingual-e5-large** to compute embeddings over the corpus. Build a FAISS index. For each chunk, retrieve top-10 candidates, filter for `sim < 0.9` (drop near-duplicates) and `source_doc != target_doc`, use top-3 valid candidates per `<retrieved>` block. This is offline data prep — the retriever is **not** trained in this stage. See `build_retrieval_pretrain_data.py` design in the data builder section below.

### Special tokens

Already in the vocabulary from Stage 0. No tokenizer modifications needed at this stage. The embeddings exist but were trained without any retrieval-format signal — Stage A is where they actually acquire retrieval semantics.

---

## Stage B: Tool-Use SFT

### What the model learns

The conversational protocol the model must master:

```
User: [Telugu question]
Model: <think>[brief reasoning about whether to search]</think>
       <search source="web">[query in Telugu or English]</search>
[Harness intercepts, retrieves, injects:]
       <retrieved>
       <doc id="d1">[passage 1]</doc>
       <doc id="d2">[passage 2]</doc>
       </retrieved>
Model: [Answer in Telugu, with <cite id="d1"/> markers on every factual claim]
```

Multi-hop variant:

```
User: [question]
Model: <search>[query 1]</search>
[retrieval]
Model: <search>[query 2 — refined based on what was retrieved]</search>
[retrieval]
Model: [grounded answer with citations]
```

Refusal variant:

```
User: [question]
Model: <search>[query]</search>
[retrieval — returns poor matches]
Model: తెలియదు. [I don't know — the retrieved sources don't contain this information.]
```

### Special tokens (final set)

```
<search source="...">  emits a query, harness intercepts
</search>
<retrieved>            harness-injected
</retrieved>
<doc id="...">         per-chunk wrapper inside retrieved block
</doc>
<cite id="..."/>       inline citation marker
<think>                optional CoT scratchpad
</think>
```

Constrained decoding at inference: when the model emits `<search ...>`, the harness pauses generation, runs retrieval, injects `<retrieved>...</retrieved>`, and resumes. Grammar-constrained decoding (logits masking) ensures the model can't produce malformed tool calls.

### Training data: synthetic generation pipeline

We need ~100–500K traces. They don't exist for Telugu — we generate them with a teacher.

**Teacher:** Claude Sonnet or Opus (strongest available). GPT-4-class is the alternative.

**Generation recipe:**

1. **Seed questions.** Sources:
   - Telugu Wikipedia entity questions ("Who is X?", "When was Y?")
   - News-driven questions from a Telugu news corpus
   - Synthetic question variations (paraphrase, multi-hop combination)
   - Adversarial questions designed to need retrieval (recent events, niche facts)
   - Questions that *don't* need retrieval (common knowledge, math, reasoning) — model must learn to skip search

2. **For each question, the teacher generates a full trace:**
   - Whether to search (sometimes the teacher emits `<no-search/>` and just answers)
   - The search query (Telugu OR English — multilingual retrieval is fine)
   - Run retrieval against a real index (we set this up, not the teacher)
   - Teacher reads retrieved passages
   - Teacher writes a grounded Telugu answer with `<cite>` markers
   - Teacher emits refusal if retrieved context is inadequate

3. **Dataset composition:**
   - 60% single-hop retrieval
   - 15% two-hop retrieval
   - 15% no-retrieval-needed (model must learn the negative case)
   - 10% refusal cases (deliberately bad retrieval → must abstain)

4. **Quality filter:**
   - Drop traces where the citation doesn't actually appear in the cited chunk (string match)
   - Drop traces where the answer contradicts retrieved evidence (NLI check)
   - Optional human review on a 1% sample

### Telugu-specific data sources for retrieval

- **Telugu Wikipedia** (~100K articles, ~200M tokens)
- **Telugu news archives** (Eenadu, Sakshi, Andhra Jyothy — if licensable; else CC-Mainichi/CulturaX)
- **Telugu government data** (telangana.gov.in, ap.gov.in PDFs)
- **Bilingual Wikipedia** (use English Wikipedia as a retrieval source, model translates context internally) — this is huge for fact coverage given Telugu Wikipedia's limited size
- **Custom user-provided sources** (textbook PDFs, domain corpora) — must work at inference time

### SFT hyperparameters

- **Full finetune** (not LoRA — at this scale, full FT is cheap and gives strictly better results)
- LR: 1e-5 (low — preserve base + Stage A capabilities)
- Batch size: small, effective 32–64 sequences
- Epochs: 2–3
- **Loss masking is critical here** (unlike Stage A): mask the `<retrieved>...</retrieved>` block during loss computation — it's harness-injected, not model output. The model must learn it does NOT generate retrieved content; the harness does. Loss applies only to model-generated tokens (queries, `<think>` blocks, answers, citations).

### Compute

- 100K traces × 2K tokens × 3 epochs = ~600M tokens of SFT
- A few days on a single A100 / H100

---

## Stage C: Faithfulness DPO

### Why

SFT teaches the *format*. It doesn't directly penalize hallucination — if the SFT teacher occasionally produces ungrounded sentences, the student inherits them. DPO directly optimizes the preference "grounded answer > hallucinated answer".

### Preference data construction

For each Stage B trace, generate two answers given the same retrieved context:

- **Chosen:** the original teacher answer (grounded, cited)
- **Rejected:** one of:
  - **Hallucinated:** teacher generates a plausible-sounding answer that includes a fact NOT in the retrieved context (we tell the teacher explicitly to do this)
  - **Miscited:** correct content but citations point to wrong chunks
  - **Confabulated when should refuse:** retrieval is bad, but teacher invents an answer instead of abstaining
  - **Over-confident:** asserts a claim that retrieved chunks only weakly support

This is auto-generated. The teacher is prompted with: "Given this question and retrieved context, write the response in two ways: one strictly grounded with citations, one that confidently makes up plausible-sounding facts not in the context."

Target: 30–100K preference pairs across the failure modes.

### DPO training

- Reference model: the Stage B SFT checkpoint
- β (KL penalty): 0.1 — keep close to SFT to preserve format
- LR: 5e-7 (DPO is sensitive; small LR)
- 1–2 epochs

Monitor: groundedness metric on a held-out eval. If it doesn't go up monotonically, β is wrong.

---

## Stage D: Inference-Time Verifier

A trained model still makes mistakes. The verifier is the last line of defense.

### Approach: small NLI head on top of the same model

Train a lightweight classifier on top of the 200M's final hidden states. Input: a generated sentence + the retrieved context. Output: `{supported, partially_supported, unsupported}`.

Training data: same as Stage C preference pairs, repurposed as a per-sentence supervision signal.

### Inference protocol

```
1. Model generates draft answer with citations.
2. Verifier scores each sentence S given retrieved chunks C.
3. If any sentence scores `unsupported`:
     a. Retry generation with a "be more cautious" system prompt
     b. If retry still fails, replace unsupported sentence with refusal phrase
     c. Or fall back to "I cannot confirm this from the retrieved sources"
4. Citations are string-matched against retrieved chunks. Any citation
   pointing to non-existent chunk text is stripped.
```

### Alternative: no separate verifier, use the model itself

Self-RAG (Asai et al.) trains the model to emit `[ISSUP]` reflection tokens inline. We can incorporate this into Stage B by adding to the training format:

```
[claim sentence] <cite id="d1"/> <issup>fully</issup>
```

Where `<issup>` ∈ {`fully`, `partially`, `none`}. The model self-grades. Cleaner — one model, one pass. But it requires more careful training data.

**Recommendation:** start with the separate verifier (simpler, more debuggable). Move to inline reflection tokens in v2 if the verifier becomes a latency bottleneck.

---

## Tool Protocol: The Harness Contract

The model only knows about *tokens*. The harness translates tokens into tool calls. Clear contract:

| Model emits | Harness does | Harness injects back |
|---|---|---|
| `<search source="web">query</search>` | Calls web search API (Brave/SerpAPI/Tavily) | `<retrieved><doc id="d1">...</doc>...</retrieved>` |
| `<search source="wikipedia">query</search>` | Queries local Telugu+English Wikipedia index | Same format |
| `<search source="custom:docs">query</search>` | Queries user-provided corpus | Same format |
| `<no-search/>` | Nothing | Nothing — model proceeds to answer |
| `</search>` without matching `<search ...>` | Decode error → reject sample | — |

**Constrained decoding** enforces the grammar. When the model is mid-tool-call, only valid tokens are allowed. The HuggingFace `transformers` library supports this via `LogitsProcessor`. We'll subclass it.

**Source allowlist** is configurable per deployment. A user can spin up the model with only `wikipedia` and a custom doc source, no web, for offline use.

---

## Retrieval Infrastructure (What Exists at Inference)

This is **not** trained — it's runtime infrastructure. The model is the consumer.

### Retriever choice

**Recommended: multilingual-e5-large.**
- Strong on Telugu (XL-Sum, MIRACL Telugu benchmark)
- 1024-dim
- Open weights
- Supports cross-lingual: Telugu query → English passage retrieval works

Alternative: **LaBSE** (smaller, faster, slightly weaker), or **BGE-M3** (newest, multilingual, dense+sparse hybrid — strongest but largest).

### Index

- **FAISS** with HNSW for local indexes (Wikipedia, custom corpora)
- **Web search via API** (Brave Search recommended — Telugu coverage; alternatives: Tavily, SerpAPI, Bing)

### Chunking

- 256–512 tokens per chunk, 64-token overlap
- Stored with `doc_id`, `source`, `url` metadata for citation backref

---

## Hallucination Defense: Multi-Layered

| Layer | Mechanism | Catches |
|---|---|---|
| **Stage A** | Continued pretraining on retrieval-augmented sequences | Sets prior that context grounds generation |
| **Stage B** | SFT with citation as part of format | Teaches the model to cite |
| **Stage B (distractor mix)** | Trains on retrieval with irrelevant chunks | Resistance to noise |
| **Stage B (refusal cases)** | Trains on bad retrieval → abstain | Calibrated abstention |
| **Stage C (DPO)** | Preference for grounded over hallucinated | Calibrates fine-grained groundedness |
| **Stage D (verifier)** | Per-sentence NLI check at decode | Catches residual hallucinations |
| **Citation string-match** | Hard postprocess | Strips fake citations |

No single layer is sufficient. Together they should bring hallucination rate to <5% on factual queries with adequate retrieval (vs. ~30–50% for a base SFT-only RAG model at this scale).

---

## Evaluation

### Intrinsic metrics

- **Groundedness:** fraction of sentences in the answer that are entailed by retrieved chunks (NLI judge, e.g., mDeBERTa-v3 multilingual NLI)
- **Citation precision:** of cited chunks, fraction that actually support the claim
- **Citation recall:** of factual claims, fraction that have a citation
- **Refusal calibration:** when retrieval is inadequate, does the model abstain? (Brier score on retrieval-quality vs. confidence)
- **Tool-call validity:** fraction of `<search>` calls that are syntactically valid + return non-empty results

### Task benchmarks (Telugu)

- **TyDi QA Telugu** — extractive QA, perfect fit for evaluating retrieval + read comprehension
- **IndicGLUE / IndicXTREME Telugu QA** subsets
- **Hand-built Telugu factual QA set** (100–500 questions, mix of needs-retrieval and doesn't-need-retrieval, with gold answers and source URLs) — we should build this; it doesn't exist publicly

### Adversarial evals

- **Trick questions** (false premises — "Why did Einstein win the Nobel for relativity?" → he didn't, won for photoelectric effect; model should correct)
- **Anachronisms** (questions about events after the index cutoff — model should say so)
- **Niche queries with no good retrieval** (model should abstain)

### A/B against alternatives

- 200M base + simple prompt-based RAG (no training) — our floor
- 200M after only Stage B (no continued pretraining, no DPO) — ablation
- Larger off-the-shelf multilingual model (Llama-3 8B, Aya-23) with same retrieval setup — our ceiling

---

## Files to Modify / Create

Stage 0 (base training) is a separate work item — covered in MEMORY.md and to be detailed in its own training script. The files below assume the base is in hand.

| File | Purpose |
|---|---|
| `train_retrieval_pretrain.py` | New. Stage A continued pretraining loop. Reformats data into retrieved+target pairs. |
| `build_retrieval_pretrain_data.py` | New. Builds the Stage A dataset using e5 retriever over the Telugu+English corpus. Outputs neighbor-lookup index; format mixing happens at the DataLoader. |
| `generate_sft_traces.py` | New. Calls teacher (Sonnet/Opus) to produce tool-use SFT traces. Quality filters built in. |
| `train_sft.py` | Existing — extend with loss masking for `<retrieved>` blocks and special token handling. |
| `generate_dpo_pairs.py` | New. Builds chosen/rejected pairs by prompting teacher for both grounded and hallucinated variants. |
| `train_dpo.py` | New. Standard DPO loop, references Stage B checkpoint. |
| `train_verifier.py` | New. Trains the NLI head on top of frozen base. |
| `inference.py` | Existing — major rewrite. Add tool-call interception, retrieval injection, verifier postprocess, grammar-constrained decoding. |
| `tokenizer-new/` | Already extended at Stage 0 — no changes needed here. |
| `retrieval/` | New directory. Index builders, retriever wrappers, web search adapters. |

---

## Risks & Open Questions

1. **Catastrophic forgetting in Stage A.** Continued pretraining at the wrong LR will erase Telugu LM ability acquired in Stage 0. Mitigation: low LR (3e-5), monitor held-out Telugu perplexity vs Stage 0 baseline, early-stop if it degrades >5%.

2. **Teacher quality bottleneck.** If Sonnet/Opus produce poor Telugu SFT traces, the student inherits the floor. Mitigation: human review on 1% sample; if quality is bad, switch to a Telugu-native teacher (e.g., Sarvam-1, OpenHathi) for the *answer* step, while keeping Claude/GPT for the *trace structure*.

3. **200M may be too small for two-hop.** Plan: train single-hop first; evaluate; only invest in multi-hop traces if single-hop reliably works.

4. **Tokenizer extension at Stage 0.** Retrieval special tokens added before base pretraining must be properly counted in vocab_size, embedding/lm_head dimensions, and the custom `morfessor_bpe_telugu_v4` loader. Verify the loader handles them as `added_tokens` (always single-token, never split by BPE).

5. **Verifier cost at inference.** A per-sentence forward pass adds latency. Profile: if it's >2× base inference, switch to a smaller verifier or inline reflection tokens.

6. **Web search rate limits & cost.** Brave/Tavily aren't free at scale. For local-only deployment, Wikipedia + custom corpora is the fallback.

7. **Multilingual citation.** If retrieved chunk is English but answer is Telugu, citation still works (refers to chunk ID), but quote-style citation needs the original chunk preserved. Design citations as chunk references, not inline quotes.

8. **No Telugu RAG benchmark exists.** We need to build one. ~200–500 questions with gold answers and source URLs. ~1 week of work.

9. **Skipping Stage A risk.** If we want a faster v0 cycle, Stage A can be skipped (the special tokens are in the vocabulary, just lack retrieval-format priors). Expected cost: SFT will need to do more work to teach the format, may converge to higher hallucination rate. Worth A/B-ing.

---

## Expected Outcomes

| Metric | Baseline (200M, prompt-RAG only) | Target (after pipeline) |
|---|---|---|
| Hallucination rate on factual QA | 30–50% | <5% |
| Citation recall | 0% (no citations) | >80% |
| Refusal precision (when retrieval is bad) | ~10% (model bluffs) | >70% |
| TyDi QA Telugu F1 | ~25–35 | 45–55 |
| Tool-call validity | n/a | >98% |

A 200M model with this training pipeline should **match or exceed an off-the-shelf 1–3B multilingual model with naive RAG** on Telugu factual QA — because the model is specifically trained for the task and language. It will not match a 7B+ model on open-ended reasoning, and we should not target that.

---

## Minimum Viable Order of Operations

Now includes Stage 0 (base training) as a prerequisite. End-to-end ~6–8 weeks.

1. **Week 0 (prerequisite):** Tokenizer extension with retrieval special tokens. Eval harness scaffold. Decide v2 vs v3 tokenizer (project memory has notes).
2. **Week 1:** Stage 0 — kick off fresh ~225M base training on B200 (3–5 epochs, mixed Telugu+English, block 4096, Tier-1 improvements per project memory). ~17–28 hours pure training; allow a week for setup + monitoring + restarts.
3. **Week 2 (parallel with Week 1):** Build retrieval infrastructure (e5 index over Telugu+English Wikipedia + Sangraha sample). Smoke-test prompt-only RAG with whatever interim base checkpoint is available, to establish floor.
4. **Week 3:** Stage A data (~2B tokens of retrieval-formatted sequences). Run light continued pretraining on the Stage 0 base. Validate Telugu perplexity didn't regress >5%.
5. **Week 4:** Generate 50K Stage B SFT traces via Claude/GPT teacher. Run SFT.
6. **Week 5:** Generate 20K DPO pairs. Run DPO. Build Telugu eval set in parallel.
7. **Week 6:** Train verifier. Integrate into inference. Evaluate end-to-end.
8. **Week 7–8:** Adversarial eval, iterate on failure modes, document.

Stage 0 + Stage A are the biggest compute items. The critical-path bottlenecks remain **data generation** (teacher quality + cost) and **Telugu eval set construction** — both should start in parallel during Stage 0 training.

---

## Appendix: Parked Alternative — Retrieval-Aware Base Pretraining

This was discussed and deferred during planning. Captured here in case we revisit.

**Premise:** rather than train a plain base in Stage 0 and bolt on retrieval format via Stage A continued pretraining, **bake retrieval awareness into base pretraining directly**. The marginal cost is just data preparation — same compute, different data format. Stage A then disappears.

**Pretraining data mix would be:**
- 50% plain text (raw documents — keep the no-context capability)
- 30% `<retrieved><doc>[related chunk]</doc></retrieved>[target]` — 1-retrieved format
- 15% multi-doc retrieved (2–3 related chunks)
- 5% distractor mix (1 relevant + 1 random chunk)

Plus 10% English data, same as the chosen plan.

**Data prep:** chunk corpus into ~512-token segments (~6M chunks total); embed all chunks with e5-large (~6h on B200); build FAISS HNSW index; for each chunk retrieve top-10 neighbors with `sim < 0.9` and `source_doc != target` filters; materialize as a neighbor lookup table (`{chunk_id: [neighbor_ids]}`, ~240MB). At training time, the DataLoader randomly selects the format and assembles the sequence on the fly — no need to materialize the full retrieval-formatted corpus on disk.

**Training:** standard causal LM loss on every token (no masking). The retrieval prior emerges from causality: target tokens attend backward to retrieved tokens, and the data distribution makes that information predictive.

**Pros:**
- Model sees retrieval format from token 0; cleaner statistical prior
- Stage A disappears (Stages B/C/D would still exist as FT)
- More efficient use of the 3B-token corpus — every token contributes to learning the retrieval semantics
- Better refusal calibration potentially baked in

**Cons (why this was parked):**
- Couples base-LM training with retrieval pipeline correctness — debugging is harder; if base goes wrong you can't isolate whether it's an LM problem or a retrieval-format problem
- Harder to compare against a plain-LM baseline (can't easily run an A/B "is the retrieval format actually helping")
- The build_retrieval_pretrain_data.py pipeline becomes blocking for Stage 0 instead of running in parallel
- Slight risk of model over-relying on context (only 50% plain text might be too low; would need to A/B this ratio)

**Decision:** keep the plain-base path (Stage 0 → A → B → C → D). The decoupling is worth the redundant Stage A compute. If Stage A fails to produce a strong retrieval prior despite the format-aware base, revisit this alternative.
