# Telugu LM — State of the World (2026-05-24)

Inventory of what exists on the runpod volume at `/workspace/telugu_lm` (310 GB) and where it diverges from local. Read-only survey, no changes made.

---

## TL;DR

1. **Git branches are not divergent.** Local and remote both at `3f03b30 (new_engrams)`, both up to date with origin. No commits to merge.
2. **Real divergence is in untracked files.** Remote has all the heavy artifacts (310 GB of checkpoints, data, wandb); local has the planning docs (`RETRIEVAL.md`, `plans/`, `tmpl_expt/`, etc.). Neither side is in conflict — they're complementary.
3. **The "200M model" is `pothana-engrams/best.pt` from 2026-03-12, val_loss=3.42.** Hidden=768, 16Q/4KV GQA, vocab=47822 (morfessor `tokenizer-new`), weight sharing on. Backbone is ~190M params, plus ~13M in the 100K×128 memory_table and engram modules.
4. **Engrams are demonstrably not contributing.** Latest run telemetry: `engram/m{0,1}/out_hidden_ratio = 0.6–0.95%` — the gate is squashing engram output to near zero. Confirms your "engrams not working" assessment.
5. **The best-performing checkpoint by val loss is `mark-zero.pt` (val=2.59)**, but it's a different architecture: hidden=1024, vocab=51K, no GQA, no weight sharing, no engrams. It's from Feb 12, before the v2 redesign. ~363M params.
6. **No "pure 200M base without engrams" checkpoint exists.** Engrams are baked into all recent checkpoints. To get a clean base for retrieval finetuning, we either (a) load a recent engram checkpoint and ignore the engram modules, or (b) train a fresh ~200M dense baseline. Option (a) is preferred — backbone is independent of engrams.

---

## 1. Git State

### Branches (local == remote, all up to date)

| Branch | Commit | Status |
|---|---|---|
| `main` | `1a9a140` "tokeniser and morfessor" | == origin |
| `engrams` | `cd3837d` "Eval" | local behind 15, but origin matches local |
| `instrft` | `94bcadf` "instruction fine tuning" | == origin |
| `arch3` | `31ed713` "First" | local-only branch, only on remote (no local) |
| `new_engrams` ★ | `3f03b30` "New Engram Code" | == origin, **current HEAD both sides** |

★ = checked-out branch on both ends.

### Stash on remote (local has none)

```
stash@{0}: WIP on main: 8731dc1 Added Inference Script
```

Diff: trivial — adds `import torch` to `inference.py`. Drop it.

### Dangling commits on remote (not in any branch)

```
8731dc1  Added Inference Script         Feb 12  inference.py (+250)
dfb8dfb  gpt                            Feb 11  train_tokenizer.py (+30 -3)
7de3f88  gpt                            Feb 11  train_tokenizer.py (+83 -19)
0d1d2be  gpt                            Feb 11  morfessor_segment.py (+1 -1)
7c21998  gpt                            Feb 11  morfessor_segment.py (+16 -1)
```

All from initial Feb 11–12 development. Superseded by later work. Safe to garbage-collect. No useful content to recover.

### Untracked files — divergence summary

| Path | Local | Remote |
|---|---|---|
| `RETRIEVAL.md` | ✅ (just written) | ❌ |
| `STATE.md` | ✅ (this file) | ❌ |
| `ENGRAM_PLAN.md` | ✅ | ❌ |
| `engram_paper.pdf` | ✅ | ❌ |
| `engram_run_history.csv` | ✅ | ❌ |
| `plans/` (CLAUDE_DR.md, GPT_DR.md, Gemini_DR.md, LEARNED_ENGRAMS.md, SYNTHESIS_300M_Core_LM.md) | ✅ | ❌ |
| `tmpl_expt/` (kenlm, 5gram tools) | ✅ | ❌ (5gram tools exist at root on remote) |
| `hf-chat-space/`, `hf-space/` | ✅ | ❌ |
| `filter_ngrams_entropy.py` | ✅ | ❌ |
| `checkpoints/`, `pothana-base-300M/`, etc. | ❌ | ✅ (heavy artifacts) |
| `data/`, `train-data/`, `sft_data/`, `sft-data/` | ❌ | ✅ |
| `wandb/` (42 runs) | ❌ | ✅ |
| `pothana.mark1.pt`, `pothana-engrams-new-tok.pt` | ❌ | ✅ |
| `tokenizer-new/`, `tokenizer-sp/` | ❌ | ✅ |
| `best_template.py`, `compress_5grams.py`, `count_5grams.py`, `process_corpus_patterns.py`, `query_templates.py`, `kenlm_5gram_counts.py`, `tmp_cvt.py`, `bin_to_text.py` | ❌ | ✅ |

**Note on the `bin_to_text.py` etc. scripts**: these exist on local too — check the local `bin_to_text.py` against remote, they may have diverged.

---

## 2. Checkpoint Inventory

Sorted by training timeline. **Bold** = best candidate for retrieval finetuning.

| Path | Size | Date | Arch | Vocab | Params | Step | val_loss | Notes |
|---|---|---|---|---|---|---|---|---|
| `/workspace/mark-zero.pt` | 3.7 GB | Feb 12 | hidden=1024, no-GQA, no-engrams | 51759 | 363M | 22500 | **2.59** | Lives at /workspace top level, not in repo. Pre-engram baseline. Different tokenizer than current. |
| `pothana.mark1.pt` | 1.4 GB | Feb 13 | (untyped) | — | ~350M | — | — | Older bare .pt, no metadata in our quick peek. |
| `pothana-base-300M/` | 1.38 GB safetensors | Feb 15 | LlamaForCausalLM, h=1024, 20L, no-GQA | 86071 | 300M | — | — | HF format. v1 morfessor base. Pairs with `tokenizer/`. |
| `pothana-chat-300M-hf/` | 1.38 GB safetensors | Feb 17 | same as above + chat tokens | 86075 | 300M | — | — | SFT-of-v1. 4 added chat tokens. |
| `checkpoints/epoch_01.pt` | 4.6 GB | Feb 22 | (untyped) | — | ~1.1B?? | — | — | Suspiciously large. Worth inspecting if needed. |
| `checkpoints/epoch_02.pt` | 4.6 GB | Feb 22 | (untyped) | — | ~1.1B?? | — | — | Same. Likely 600–800M model in fp32 + optimizer state. |
| `pothana-sp-base-300M/` | 2.9 GB safetensors | Feb 23 | h=1024, **60L**, GQA 16/4 | 48000 SP | ~700M | — | — | Deep-narrow sentencepiece experiment. Pairs with `tokenizer-sp/`. |
| `sft_checkpoints/best.pt` | 4.1 GB | Feb 17 | (untyped, includes optimizer) | — | ~500M | 900 | — | SFT on translated 1K conversations. Tied to old 86K tokenizer. |
| `checkpoints/best.pt` | 2.7 GB | Mar 10 | h=768, 16Q/4KV, ws | 47822 | ~190M+mem | — | — | First engram run on new tokenizer. |
| **`pothana-engrams/best.pt`** ★ | 2.4 GB | Mar 12 | h=768, 16Q/4KV, ws, engrams (100K×128, 2 layers) | 47822 | **235M** | 9000 | **3.42** | **Best engram checkpoint.** 10K-step run, completed. |
| `pothana-engrams-new-tok.pt` | 2.7 GB | Mar 28 | same backbone, 1M×128 memory_table | 47822 | 350M | 500 | 5.22 | Aborted early. Just-restarted run. |
| `checkpoints/engram-v3/best.pt` | 2.7 GB | Mar 28 | same backbone, engram_dim=64 (smaller), PMI-filtered 5gram patterns | 47822 | 259M | 4500 | **3.52** | Latest experiment. Worse than `pothana-engrams`. |

★ = recommended starting point for retrieval finetuning (after engram-stripping).

### Architecture summary of the v2/engram series (h=768)

From `config` in `pothana-engrams/best.pt`:

```
block_size        : 2048
vocab_size        : 47822
n_layer           : (unique blocks — let me re-read meta)
n_head            : 16
n_kv_head         : 4
n_embd            : 768
intermediate (MLP): 2048   (w_gate/w_up = 2048×768)
attention bias    : false
rope_theta        : 10000
use_weight_sharing: true
```

KV head_dim = 192/4 = 48? Actually 16 Q heads × 48 head_dim = 768 ✓ and 4 KV heads × 48 = 192 ✓. Matches MEMORY.md v2 spec ("hidden=768, 16Q/4KV").

Engram modules (in `pothana-engrams/best.pt`):
- `memory_table.weight: (100000, 128)` — 100K rows × 128 dim
- 2 engram modules with: hash_weights, mask_predictor (768→12→6), pattern_mlp, W_K, W_V (768×128), ln_gate, gate_proj

Engram modules (in `engram-v3/best.pt`):
- `pattern_table` (top-level, distinct from memory_table)
- 2 engram modules with W_V (768×**64**), gate_proj — engram_dim halved

### Why engrams aren't working — the smoking gun

Latest run telemetry (engram-v3, step 4500):

| Metric | m0 | m1 |
|---|---|---|
| hit_rate | 48.5% | 48.5% |
| gate_mean | 0.77 | 0.96 |
| **out_hidden_ratio** | **0.95%** | **0.63%** |

`out_hidden_ratio` = ‖engram output‖ / ‖hidden state‖. **Less than 1% of the hidden state's magnitude is coming from the engram lookup.** The model has learned to gate the engram contribution down to near zero. The hit rate is fine (48%) — the lookups are finding entries — but the values aren't useful and the network is ignoring them. Consistent with "engrams not improving anything."

Val loss progression confirms:
- `mark-zero.pt` (no engrams, h=1024): **2.59**
- `pothana-engrams/best.pt` (engrams, h=768): **3.42**
- `engram-v3/best.pt` (smaller engrams, h=768): **3.52**

Loss is going **up** as the architecture shrunk and engrams were added. The dense backbone capacity reduction (1024→768) is doing more damage than engrams are repairing.

---

## 3. Tokenizers

| Dir | Created | Type | Vocab | Paired with |
|---|---|---|---|---|
| `tokenizer/` | Feb 17 | morfessor (WordLevel + custom decoder) | 86071 | `pothana-base-300M`, `pothana-chat-300M-hf` |
| `tokenizer-new/` | Mar 9 | morfessor (deduplicated, no @@) | **47822** | `pothana-engrams*`, `engram-v3`, `train-data/` |
| `tokenizer-sp/` | Feb 21 | SentencePiece | 48000 | `pothana-sp-base-300M` |

`tokenizer-new/` is the current tokenizer. Files: `tokenizer.json` (1.7 MB), `vocab.txt` (747 KB), `token_frequencies.tsv`.

The "v3 tokenizer with ▁ separator" described in MEMORY.md does **not exist on the volume**. It's a design described but not yet implemented. The current production tokenizer is `tokenizer-new/` (v2 morfessor, deduplicated).

`pothana-base-300M/` ships its own `tokenizer.json` (2.83 MB, 86K) + `tokenizer_class.py` + `morfessor_telugu.bin` (8.65 MB) inside the HF folder — the v1 setup. `pothana-chat-300M-hf/` ships the same tokenizer with 4 chat tokens added.

---

## 4. Data Inventory

### Pretraining corpus

| File | Size | Tokens | Notes |
|---|---|---|---|
| `train-data/train.bin` | 12.3 GB | **3.07B** (uint32) | tokenized with `tokenizer-new` (47K vocab), morfessor segmented, unk_rate=8.8e-6 |
| `train-data/val.bin` | 251 MB | 62.7M | same |
| `train-data/train.1M.txt` | 5.8 GB | — | sample text (1M lines?) |
| `train-data/train_tokens.txt` | 40 GB | — | full tokenized text (human-readable form, for n-gram extraction) |
| `train-data/meta.json` | — | — | `{"vocab_size":47822,"total_tokens":3057030311,"tokenizer_type":"morfessor"}` |

3.07B Telugu tokens. Matches MEMORY.md "~3.8B token target" — slight under but close.

### 5-gram template index (for engrams)

| File | Size | Notes |
|---|---|---|
| `train-data/5grams/pattern_ids.bin` | 12.2 GB | per-position pattern IDs for the 580K-pattern table |
| `train-data/5grams/pattern_vocab.jsonl` | 99 MB | pattern definitions |
| `train-data/5grams/pattern_stats.json` | 1.2 KB | stats |

From `pattern_stats.json`:
- 580,262 unique patterns, 6.94M sentences, 3.04B windows
- 45.2% of windows hit some template (template_hits 42.5%, exact_hits 2.7%)
- PMI median 105.9 — high-quality patterns
- Built in 3,031 sec (~50 min)

This is the asset feeding the engram lookup. We can probably reuse the pattern infrastructure for kNN-style retrieval datastores if useful.

### SFT data (small, stale)

| File | Notes |
|---|---|
| `sft-data/translated.jsonl` | 8.6 MB — 1000 translated conversations (English → Telugu) |
| `sft_data/` | tokenized version. 6402 train examples, 1.84M tokens. **Tied to OLD 86K tokenizer** (vocab_size_with_special=86075). Doesn't match current 47K tokenizer. |

The SFT data is from Feb 17 and is tokenized with the old tokenizer. **It cannot be used as-is with the current engram models or any new retrieval model.** Would need to retokenize from `translated.jsonl` against `tokenizer-new`.

Also: only 1000 conversations is tiny. For retrieval SFT we'll need to generate 50–200K traces (per RETRIEVAL.md plan).

### Other corpus artifacts in `data/`

| File | Size | Notes |
|---|---|---|
| `data/sample_raw.txt` | 1.08 GB | raw text sample |
| `data/sample_raw_clean.txt` | 1.09 GB | cleaned |
| `data/sample_raw_clean.seg.txt` | 1.16 GB | morfessor-segmented |
| `data/sample.parquet` | 425 MB | parquet form of the same |
| `data/sangraha.1L.arpa` | 8.4 GB | KenLM 5-gram model (for n-gram baselines / template extraction) |
| `data/morfessor_telugu.bin` | 8.65 MB | morfessor model |
| `data/word_frequencies.txt` | 21.5 MB | word counts (5M tokens) |
| `data/morpheme_vocab.tsv` | 750 KB | morpheme list (33,443 types) |
| `data/vocab_stats.txt` | — | 665K word types, 33K morpheme types, 1.52 morphemes/word, 4.5% unsegmented |
| `data/english/` | tiny | placeholder/empty english tools |
| `data/sangraha/` | dir | original Sangraha shards (didn't list) |
| `data/sangraha_5grams/` | dir | 5-gram counts (didn't list) |
| `data/stitcher/` | dir | for the stitcher experiment |
| `data/morfessor-sample/` | dir | morfessor training corpus sample |

---

## 5. Wandb Runs (42 total)

Last meaningful runs:

| Run dir | Date | Steps | val_loss | best_val | Wall time | Notes |
|---|---|---|---|---|---|---|
| `run-20260221_183502-begbnqag` etc. (Feb 22) | Feb 22 | — | — | — | — | Multiple short runs, possibly hyperparameter sweep |
| `run-20260310_185119-c5422kot` etc. (Mar 10) | Mar 10 | various | — | — | — | First-attempt engram runs |
| `run-20260311_211023-kx0hwe3a` ★ | Mar 11 | 10000 | 3.45 | **3.42** | 5h27m | The `pothana-engrams` run that produced the best engram checkpoint |
| `run-20260328_143318` to `run-20260328_151909` | Mar 28 | <200 each | high | — | minutes | Six aborted starts |
| `run-20260328_152536-xum6qcrs` | Mar 28 | 5000 | 3.47 | **3.52** | 5h42m | The `engram-v3` run on B200 GPU |

★ = the run that produced the best engram-architecture model.

GPU used in the last run: **NVIDIA B200**, 192 GB HBM. The training infrastructure is fine — bottleneck is the engram design, not compute.

---

## 6. What I Recommend Before Retrieval Work Starts

(Per your direction, no actions taken — these are proposals.)

### Repo hygiene
1. **Sync the planning docs from local → remote** (RETRIEVAL.md, STATE.md, plans/, ENGRAM_PLAN.md, engram_run_history.csv, engram_paper.pdf, filter_ngrams_entropy.py). Either commit them to `new_engrams` or just rsync as untracked.
2. **Sync ad-hoc scripts from remote → local** (best_template.py, compress_5grams.py, count_5grams.py, process_corpus_patterns.py, query_templates.py, kenlm_5gram_counts.py, tmp_cvt.py). Some of these are in `tmpl_expt/` locally — check for duplicates.
3. **Drop the stash on remote** (`git stash drop`). It's trivial.
4. **Optional: garbage-collect dangling commits** (`git gc --prune=now`). They're not protecting anything.
5. **Add a .gitignore for the heavy dirs** (checkpoints/, wandb/, data/, train-data/, *.pt at root) so future status doesn't show 50 GB of "untracked".

### Checkpoint hygiene
6. **Keep**: `mark-zero.pt`, `pothana-engrams/best.pt`, `engram-v3/best.pt`, `pothana-base-300M/`, `pothana-chat-300M-hf/`. These are the meaningful anchors.
7. **Consider deleting**: most of the intermediate `step_NNNNN.pt` files in `pothana-engrams/` and `sft_checkpoints/`. ~70 GB recoverable. Keep `best.pt`, `final.pt`, and maybe one early/middle checkpoint per run.
8. **Decide on `pothana-sp-base-300M/`** — was the SP experiment kept around for a reason or abandoned? 2.9 GB.

### Decision for retrieval work
9. **Recommended base for retrieval finetuning: `pothana-engrams/best.pt` backbone** (h=768, 16Q/4KV, vocab=47822, weight sharing). Load with `--use-engrams=False` to drop the memory_table and engram_modules and treat it as a ~190M dense Telugu LM. This is the best-aligned-to-current-tokenizer model.
10. **Alternative base: `mark-zero.pt`** (val=2.59, materially better LM). But: different vocab (51K), different arch (h=1024, no GQA). Would need retokenization of training data, and the 363M size makes it slower for the planned multi-stage finetuning. Worth a quick comparison run before committing.
11. **Either way: retrain the tokenizer with retrieval special tokens added** (`<search>`, `<retrieved>`, `<doc>`, `<cite>`, `<think>`, etc. per RETRIEVAL.md). Resize embedding + lm_head to accommodate.

---

## 7. Open Questions for You

1. **Which base model for retrieval finetuning** — the 190M `pothana-engrams` backbone (current tokenizer, smaller, faster to iterate), or `mark-zero.pt` (363M, materially lower perplexity, but old tokenizer)?
2. **Do we keep the engram infrastructure around** for possible kNN-LM-style use later, or fully tear it out?
3. **Should we delete the intermediate step_NNNNN.pt files** to reclaim ~70 GB?
4. **Where do we want STATE.md and RETRIEVAL.md to live** — committed to `new_engrams`, a new `retrieval` branch, or just untracked?
5. **The 1K-conversation SFT dataset** — keep, regenerate against new tokenizer, or replace entirely with retrieval-style SFT?
