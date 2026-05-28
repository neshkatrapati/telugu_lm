# Base v2 — Prep Complete

All pre-training work is done on the CPU box. Volume is shared with the B200 (`/workspace` is the same network mount), so the B200 sees everything immediately on attach — no transfer needed.

## What's on the volume now

| Path | Size | Purpose |
|---|---|---|
| `/workspace/telugu_lm/tokenizer-new-v5/` | 4.4 MB | Extended tokenizer, vocab 47831, +9 retrieval special tokens |
| `/workspace/telugu_lm/train_gpt_v2.py` | 80.7 KB | Patched trainer (untie / z-loss / QK-norm / block_size + rope_theta + warmup CLI) |
| `/workspace/telugu_lm/train-data-v2/train.bin` | 13.48 GB | 3.37B tokens — 91.2% Telugu + 8.8% English |
| `/workspace/telugu_lm/train-data-v2/val.bin` | 263 MB | 65.7M tokens — Telugu + English |
| `/workspace/telugu_lm/train-data-v2/meta.json` | 1 KB | Vocab=47831, tokenizer-new-v5 |
| `/workspace/telugu_lm/train-data-v2/english.bin` | 1.20 GB | Standalone English data (kept for reproducibility) |
| `/workspace/telugu_lm/train-data-v2/english.meta.json` | 0.3 KB | English prep metadata |
| `/workspace/telugu_lm/BASE_V2_PATCH.md` | 16 KB | Patch reference doc (now mostly historical) |
| `/workspace/telugu_lm/extend_tokenizer.py` | 2.8 KB | Tokenizer-extension script (already run) |
| `/workspace/telugu_lm/prep_english.py` | 5.5 KB | English download + tokenize script (already run) |
| `/workspace/telugu_lm/mix_v2.py` | 6.5 KB | Telugu+English mix script (already run) |

## Smoke-test status

CPU smoke test of `train_gpt_v2.py` with all Tier-1 flags enabled passed:

- Builds at **222.1M params** with untied embeddings (185.4M with tied — confirms +36.7M from untying)
- QK-norm modules present with correct shapes `(48,)` (head_dim)
- Z-loss contributes ~0.012 to total loss at init (small, as designed)
- Forward + backward produce finite loss + gradients
- End-to-end forward on a real window from `train-data-v2/train.bin` returns loss=10.93 (matches `ln(47831)=10.77` plus small noise, expected at random init)

## Pre-launch checklist (do these on the B200)

1. **Verify the volume mount.** Confirm `/workspace/telugu_lm/train_gpt_v2.py` is visible and timestamps match what's reported above.
2. **Activate conda env.** `source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate tlm`. Verify `python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` shows B200.
3. **GPU smoke test (5 min).** Run the small smoke recipe at the bottom of `BASE_V2_PATCH.md` with `--max-steps 200` to verify B200 tput is reasonable (expect ~80-100K tok/sec at block 4096) and no CUDA/compile issues.
4. **Kick off full Base v2 run.**

## The full Base v2 launch command

```bash
cd /workspace/telugu_lm
python train_gpt_v2.py train \
    --data train-data-v2 \
    --tokenizer tokenizer-new-v5 \
    --block-size 4096 \
    --rope-theta 500000 \
    --n-layer 24 \
    --n-embd 768 \
    --n-kv-head 4 \
    --untie-embeddings \
    --use-z-loss \
    --use-qk-norm \
    --batch-size 64 \
    --grad-accum 8 \
    --lr 5e-4 \
    --warmup-steps 3000 \
    --max-steps 15000 \
    --lr-schedule wsd \
    --wsd-stable-frac 0.7 \
    --wsd-decay-frac 0.2 \
    --grad-checkpoint \
    --save-dir checkpoints/base-v2 \
    --save-interval 1000 \
    --wandb pothana-base-v2 \
    --wandb-name base-v2-r1
```

Math behind this:
- Effective batch = 64 × 8 = 512 sequences
- Tokens per step = 512 × 4096 = 2,097,152
- 15,000 steps × 2.1M tok = ~31.5B effective tokens
- 31.5B / 3.37B = **~9.3 epochs** (more than the 3–5 originally planned — comfortable margin; can early-stop at 9–10K if val loss plateaus)

If you want exactly 3 epochs: `--max-steps 5000`. For 5 epochs: `--max-steps 8000`.

## What to monitor

- **train/loss** smooth decreasing — first ~3000 steps will be warmup-LR-ramping, loss drops fast then
- **val/loss** every 500 steps — target trajectory: starts at ~10.7, drop to ~5 by step 1000, ~3.5 by step 5000, ~2.8 by step 15000 (rough estimates)
- **tok/sec** stable around 80–100K on B200 at 4K context
- **CUDA mem** — at batch 64 × 4096 × untied lm_head + grad-ckpt, should fit in 192 GB B200 comfortably (estimate ~40–50 GB peak with optimizer states)
- **GPU util** above 90% sustained

## Things that didn't make this prep (deliberate)

- **No retrieval format sprinkle in base data.** Decided in earlier conversation — Stage A handles that later.
- **No code/math mix.** Out of scope for Base v2 (Tier-2 items).
- **No data quality filtering.** Sangraha was deemed already-good.
- **No tokenizer fertility re-audit on Telugu** — the v5 just adds 9 tokens, doesn't change Telugu encoding.

## Risks worth knowing about

1. **English fertility is ~1.81 tokens/word** (probed on Wiki samples) — about 2× worse than a dedicated English BPE. Not catastrophic but means the model "sees" English at lower density. Acceptable for 8.8% of the corpus; would be a problem if English were the dominant language.

2. **The Telugu portion of `train-data-v2/train.bin` is a verbatim copy of the original `train-data/train.bin`** (tokenized with the 47822-vocab old tokenizer). Token IDs 0–47821 are identical between v4 and v5 tokenizers, so this is safe — but the **9 retrieval special tokens (IDs 47822–47830) never appear in the Telugu data**. They'll appear in neither Telugu nor English text naturally. Their embedding rows will sit at random init through all of Base v2 training, then start learning during Stage A retrieval continued-pretrain. Expected and noted.

3. **Telugu/English boundary** in train.bin (token offset 3,073,162,645) is a single transition. The `MemmapDataset.get_batch` does pure random uniform sampling over the file, so windows that straddle the boundary are extremely rare (1 boundary in ~3.37B tokens). Not a practical issue.

4. **Engram CLI flags are still present** in `train_gpt_v2.py` but not used (no `--use-engrams`). Old engram code paths inert. If someone passes `--use-engrams` they'll re-enable the old machinery — don't.

5. **`block_size` doesn't affect data prep** (data is just a flat memmap), only the trainer's sampling window. So the same `train.bin` would work if you ever want to try block_size=8192 later.

## How to verify B200 sees the same files

```bash
# Compare md5 of v2 trainer between CPU prep box and B200
md5sum /workspace/telugu_lm/train_gpt_v2.py
# Expected: same value across machines (network volume = identical bytes)
```

## Recovery / restart

- Checkpoints saved every 1000 steps to `checkpoints/base-v2/step_NNNNN.pt`
- To resume: add `--resume checkpoints/base-v2/step_XXXX.pt` to the launch command
- Best-val checkpoint auto-saved as `checkpoints/base-v2/best.pt`
- Final checkpoint as `checkpoints/base-v2/final.pt`

## Next deliverable (after Base v2 trains)

Per the `RETRIEVAL.md` pipeline, the next item is **Stage A: light retrieval-aware continued pretraining**. Data builder for that (`build_retrieval_pretrain_data.py`) is not yet written — that's the next major work item, separate from this prep. It needs:
- multilingual-e5-large embeddings over the Telugu+English corpus (~6h on B200)
- FAISS index + neighbor lookup
- DataLoader that on-the-fly formats sequences with `<retrieved>...</retrieved>` blocks

Not blocking — can be built while Base v2 is training.
