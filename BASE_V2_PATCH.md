# Base v2 — Patch Spec for `train_gpt.py`

Surgical edits to the existing `train_gpt.py` to enable Tier-1 improvements:

- `block_size = 4096`, `rope_theta = 500000`
- Untied embeddings (`lm_head` separate from `wte`)
- Z-loss (output normalization)
- QK-norm in attention
- New LR / batch / warmup defaults
- Engrams disabled (just don't pass `--use-engrams`)

All edits are additive and gated by new config flags — old behavior is the default. New training enables flags at the CLI.

The actual tokenizer extension (adding `<search>`, `<retrieved>`, etc.) is a separate one-time script, included at the bottom.

---

## Edit 1 — `GPTConfig` (add Tier-1 flags + new defaults)

**File:** `train_gpt.py`, around line 61 (`@dataclass class GPTConfig`)

**Add these fields** (preserve all existing fields; defaults are conservative so old training scripts still work):

```python
@dataclass
class GPTConfig:
    """LLaMA-style model configuration."""
    block_size: int = 2048
    vocab_size: int = 0
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    use_weight_sharing: bool = True

    # === Base v2 — Tier-1 improvements ===
    tie_embeddings: bool = True       # set False for untied lm_head (Base v2)
    use_z_loss: bool = False          # set True for output normalization (Base v2)
    z_loss_weight: float = 1e-4       # λ × log(Z)² penalty
    use_qk_norm: bool = False         # set True for QK-norm in attention (Base v2)

    # Engrams (kept for backwards compat — Base v2 won't enable them)
    use_engrams: bool = False
    ...
```

---

## Edit 2 — `CausalSelfAttention` (add QK-norm)

**File:** `train_gpt.py`, around line 490 (`class CausalSelfAttention`)

Modify `__init__` to add QK-norm modules conditionally, and `forward` to apply them before RoPE.

**In `__init__`, after the existing projections** (around line 508, after `self.resid_dropout = ...`):

```python
            # QK-norm (Llama 3.1 / Cosmos) — RMSNorm on Q and K before RoPE
            self.use_qk_norm = config.use_qk_norm
            if self.use_qk_norm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
```

**In `forward`, after the Q/K/V projections but BEFORE `apply_rotary_emb`** (around line 517):

```python
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

            # QK-norm — applied to each head dim before RoPE
            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            # Apply RoPE to Q and K
            q, k = apply_rotary_emb(q, k, freqs_cis)
```

Note: QK-norm applies to the head-dim axis. RMSNorm broadcasts over the leading dims (B, n_head, T) — works for both Q and K despite different head counts.

---

## Edit 3 — `GPT.__init__` (untie embeddings conditionally)

**File:** `train_gpt.py`, around line 663 — the existing line:

```python
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying
            self.transformer.wte.weight = self.lm_head.weight
```

**Replace with:**

```python
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying (default: tied; Base v2: untied for capacity)
            if config.tie_embeddings:
                self.transformer.wte.weight = self.lm_head.weight
```

Untying adds ~36M params (vocab × n_embd = 47831 × 768 ≈ 36.7M) and means `lm_head.weight` is independently initialized + trained. The `_init_weights` call already handles it correctly (it inits all Linear weights normally).

---

## Edit 4 — `GPT.forward` (add z-loss)

**File:** `train_gpt.py`, around line 765 — existing block:

```python
            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

            return logits, loss
```

**Replace the `if targets is not None:` branch with:**

```python
            if targets is not None:
                logits = self.lm_head(x)
                # Compute CE loss
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )
                # Z-loss: λ × E[log(sum(exp(logits)))²]
                # Prevents logit drift, used by PaLM / Gemini / Llama-3
                if self.config.use_z_loss:
                    logsumexp = torch.logsumexp(logits.float(), dim=-1)  # (B, T)
                    z_loss = self.config.z_loss_weight * (logsumexp ** 2).mean()
                    loss = ce_loss + z_loss
                else:
                    loss = ce_loss
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
```

Both losses are now logged separately in z-loss mode. If you want to monitor z-loss magnitude in wandb, also return it — but for now, folding it into the total `loss` is enough since it's small (~0.01 contribution).

---

## Edit 5 — `TrainConfig` defaults for Base v2

**File:** `train_gpt.py`, around line 109 (`class TrainConfig`)

These are *defaults* only — the CLI in `main()` overrides them. But updating them keeps "run with no flags" sane.

```python
@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Batch — Base v2: effective 512 (was 128 in defaults; latest engram run used 256)
    micro_batch_size: int = 64
    gradient_accumulation_steps: int = 8

    # Optimizer — Base v2: peak 5e-4 (was 3e-4)
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule — Base v2: warmup 3000 (was 500)
    warmup_steps: int = 3000
    max_steps: int = 45000           # adjust based on epochs × tokens / effective batch
    lr_decay_steps: int = 45000
    lr_schedule: str = "wsd"
    wsd_stable_frac: float = 0.7
    wsd_decay_frac: float = 0.2

    # rest unchanged...
```

Math check on `max_steps` for Base v2:
- 3 epochs × 3.4B tokens = 10.2B effective tokens
- Effective batch 512 × seq_len 4096 = 2,097,152 tokens/step
- 10.2B / 2.1M ≈ **4860 steps per epoch × 3 ≈ 14,500 steps total** (3 epochs)
- For 5 epochs: ~24,300 steps

So `max_steps=15000` for 3 epochs, `max_steps=24000` for 5 epochs. Adjust at CLI invocation.

---

## Edit 6 — CLI flags in `main()`

**File:** `train_gpt.py`, in `main()` around line 1529.

Add these argparse flags (preserving existing ones):

```python
    parser_train.add_argument("--untie-embeddings", action="store_true",
                              help="Untie lm_head from wte (Base v2)")
    parser_train.add_argument("--use-z-loss", action="store_true",
                              help="Add z-loss term (Base v2)")
    parser_train.add_argument("--z-loss-weight", type=float, default=1e-4)
    parser_train.add_argument("--use-qk-norm", action="store_true",
                              help="Apply QK-norm in attention (Base v2)")
    parser_train.add_argument("--rope-theta", type=float, default=None,
                              help="Override rope_theta (e.g., 500000 for long context)")
```

Then in the code that constructs `GPTConfig` from `args` (find the existing `GPTConfig(...)` call inside `main`):

```python
    config_kwargs = dict(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta if args.rope_theta else 10000.0,
        use_weight_sharing=args.use_weight_sharing,
        tie_embeddings=not args.untie_embeddings,
        use_z_loss=args.use_z_loss,
        z_loss_weight=args.z_loss_weight,
        use_qk_norm=args.use_qk_norm,
        # engram fields unchanged...
    )
    config = GPTConfig(**config_kwargs)
```

(Adjust to match the actual variable names in your existing `main()`.)

---

## Launch command for Base v2

After tokenizer is extended (see script below) and the patches are applied:

```bash
python train_gpt.py train \
    --data train-data \
    --tokenizer tokenizer-new \
    --block-size 4096 \
    --rope-theta 500000 \
    --n-layer 24 \
    --n-head 16 \
    --n-kv-head 4 \
    --n-embd 768 \
    --use-weight-sharing \
    --untie-embeddings \
    --use-z-loss \
    --use-qk-norm \
    --batch-size 64 \
    --grad-accum 8 \
    --lr 5e-4 \
    --min-lr 5e-5 \
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

(Adjust flag names to match your actual argparse — `--batch-size` may be `--micro-batch-size`, etc.)

Expected:
- ~225M params (190M from h=768 backbone + 36M untied lm_head)
- ~15K steps × ~30s/step on B200 = ~13 hours for 3 epochs
- val loss target: 2.8–3.0 (vs 3.42 at 0.85 epochs with engrams; pure-Telugu corpus only, no English mix yet)

If adding 10% English (recommended), do that data-prep work first — see "Follow-ups" at the bottom.

---

## Tokenizer Extension Script — `extend_tokenizer.py`

One-time script to add retrieval special tokens to `tokenizer-new/`. Adds 9 tokens, vocab grows 47822 → 47831.

```python
#!/usr/bin/env python3
"""
Extend the morfessor_bpe_telugu_v4 tokenizer with retrieval special tokens.

Run once before Base v2 training:
    python extend_tokenizer.py --in tokenizer-new --out tokenizer-new-v5

Adds 9 tokens: <search>, </search>, <retrieved>, </retrieved>,
                <doc>, </doc>, <cite>, <think>, </think>
"""

import json
import argparse
import shutil
from pathlib import Path


RETRIEVAL_TOKENS = [
    "<search>",
    "</search>",
    "<retrieved>",
    "</retrieved>",
    "<doc>",
    "</doc>",
    "<cite>",
    "<think>",
    "</think>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True,
                        help="Input tokenizer dir (e.g., tokenizer-new)")
    parser.add_argument("--out", dest="out_dir", required=True,
                        help="Output tokenizer dir (will be created)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    # Copy everything over first
    if out_dir.exists():
        raise SystemExit(f"output dir {out_dir} already exists — refusing to overwrite")
    shutil.copytree(in_dir, out_dir)

    # Load and patch tokenizer.json (custom format: morfessor_bpe_telugu_v4)
    tok_path = out_dir / "tokenizer.json"
    with open(tok_path) as f:
        tok = json.load(f)

    assert tok["type"] == "morfessor_bpe_telugu_v4", f"unexpected type: {tok['type']}"

    token_to_id = tok["token_to_id"]
    current_max = max(token_to_id.values())
    print(f"current vocab_size: {tok['vocab_size']}, max id: {current_max}")

    added = []
    next_id = current_max + 1
    for t in RETRIEVAL_TOKENS:
        if t in token_to_id:
            print(f"  already present: {t} → {token_to_id[t]}")
            continue
        token_to_id[t] = next_id
        added.append((t, next_id))
        next_id += 1

    # Update top-level vocab_size and special_tokens
    tok["vocab_size"] = max(token_to_id.values()) + 1
    if "special_tokens" not in tok:
        tok["special_tokens"] = {}
    for t, i in added:
        tok["special_tokens"][t] = i

    # Save
    with open(tok_path, "w") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)

    # Also patch vocab.txt if it exists
    vocab_txt = out_dir / "vocab.txt"
    if vocab_txt.exists():
        with open(vocab_txt, "a") as f:
            for t, _ in added:
                f.write(t + "\n")

    print(f"\nAdded {len(added)} tokens:")
    for t, i in added:
        print(f"  {i}: {t}")
    print(f"\nNew vocab_size: {tok['vocab_size']}")
    print(f"Output: {out_dir}")
    print("\nNext steps:")
    print("  1. Verify with your custom tokenizer loader that the new tokens encode as single IDs")
    print(f"  2. Update train.bin meta.json or use --vocab-size {tok['vocab_size']} at training time")


if __name__ == "__main__":
    main()
```

After running this, the new tokenizer is at `tokenizer-new-v5/` (or whatever you choose). Update the training launch to point at it. Note: existing `train.bin` was tokenized with the old 47822-vocab tokenizer — the new special tokens won't appear in the data, but that's fine; they'll be unused during base pretraining and only used during retrieval FT. The IDs are reserved and the embedding rows will be there.

**Important caveat:** since `train.bin` was tokenized without these special tokens, no embedding signal flows through them during base training. They'll be left at their random init. For Stage A continued pretraining (which DOES use the special tokens), those embeddings will start training from scratch — that's fine but worth being aware of.

If you want the special tokens to start with reasonable embeddings even before Stage A, two options:
1. Sprinkle ~1% of base-training sequences with synthetic `<retrieved>...</retrieved>[target]` examples. Cheap, no extra data needed (just self-retrieval).
2. After base training, initialize each special token's embedding as the average of its neighbors' embeddings (e.g., `<retrieved>` ≈ mean of "<", "retrieved", ">" if those subwords exist). One-line tweak.

Option 1 is what the appendix in RETRIEVAL.md called the "5% retrieval format sprinkle" — recommended if you have data-pipeline bandwidth.

---

## Follow-ups (separate work items, not in this patch)

These are needed before Base v2 actually launches but are out of scope for the train_gpt.py patch:

1. **English data prep.** Tokenize ~300M tokens of English Wikipedia + FineWeb-Edu using the same morfessor_bpe_telugu_v4 tokenizer (it should fall back to BPE+char for English). Append to or interleave with `train.bin`. Update `meta.json` for total token count.

2. **Eval harness.** Build `eval/` with: perplexity script, IndicGLUE Telugu subsets, TyDi QA Telugu, hand-built 50-question probe set. Run before training starts to establish baseline on existing checkpoints (mark-zero.pt as reference point).

3. **Code+math mix** (optional Tier-2). 3% Python code from The Stack, 2% math (GSM-Hindi / MATH translated). Add to data mix after English is in.

4. **Wandb run naming convention.** Suggest: `base-v2-{tag}` where tag describes the differentiator (e.g., `base-v2-r1`, `base-v2-en10`, `base-v2-untied-only`).

5. **Verify B200 throughput with new settings.** With block_size=4096 (vs 2048) and untied lm_head, you'll see ~30–40% lower tok/sec. Plan compute accordingly.

---

## Smoke test before the full run

Before kicking off the 15K-step run, do a 200-step smoke test with the same config to verify:

```bash
python train_gpt.py train \
    --data train-data --tokenizer tokenizer-new-v5 \
    --block-size 4096 --rope-theta 500000 \
    --untie-embeddings --use-z-loss --use-qk-norm \
    --batch-size 32 --grad-accum 4 \
    --lr 5e-4 --warmup-steps 100 --max-steps 200 \
    --save-dir /tmp/smoke --no-wandb
```

Verify:
- Memory footprint fits (B200 has 192 GB — should be fine)
- Loss is finite and decreasing
- tok/sec is reasonable (~80–100K on B200 at 4K context)
- No NaN spikes from z-loss interaction

If smoke passes, kick off the full run.
