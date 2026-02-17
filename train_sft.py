#!/usr/bin/env python3
"""
Telugu LLaMA — Supervised Fine-Tuning (SFT)
=============================================
Full fine-tune the pretrained 300M Pothana Base model on chat-formatted
instruction data produced by build_sft_dataset.py.

Key differences from pretraining (train_gpt.py):
  - Loads pretrained checkpoint and resizes embeddings for new special tokens
  - Uses per-example data loading with variable-length sequences and padding
  - Loss masking: CE only on assistant response tokens (mask=1)
  - Lower learning rate (2e-5) and weight decay (0.01) to preserve knowledge
  - Epoch-based training (3-5 epochs) instead of step-based

Input:  SFT dataset from build_sft_dataset.py
        (train.bin, train.mask.bin, train.offsets.json, meta.json)
Output: Fine-tuned checkpoints in save directory

Usage:
    python train_sft.py \\
        --checkpoint ./checkpoints/best.pt \\
        --data ./sft_data \\
        --tokenizer ./tokenizer \\
        --morfessor ./morfessor_telugu.bin \\
        --save-dir ./sft_checkpoints \\
        --epochs 3 \\
        --wandb pothana-sft
"""

import os
import re
import sys
import math
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")


# ============================================================================
# SFT Training Configuration
# ============================================================================
@dataclass
class SFTConfig:
    """SFT hyperparameters — conservative to preserve pretrained knowledge."""
    # Batch
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 2  # effective batch = 16

    # Optimizer
    learning_rate: float = 2e-5       # 10x lower than pretraining
    min_lr: float = 2e-6              # 10% of max LR
    weight_decay: float = 0.01        # lighter than pretraining (0.1)
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 50
    max_epochs: int = 3

    # Logging & saving
    log_interval: int = 5
    eval_interval: int = 50            # steps between eval
    eval_steps: int = 0                # 0 = eval on full val set
    save_interval: int = 100
    save_dir: str = "./sft_checkpoints"

    # Performance
    compile_model: bool = True
    dtype: str = "bfloat16"

    # Logging
    wandb_project: str = ""
    wandb_run_name: str = ""


# ============================================================================
# SFT Dataset
# ============================================================================
class SFTDataset:
    """Dataset for SFT training with variable-length examples and loss masks.

    Loads binary files produced by build_sft_dataset.py:
      - {split}.bin         — uint32 token IDs (flat, all examples concatenated)
      - {split}.mask.bin    — uint8 loss masks (parallel to token IDs)
      - {split}.offsets.json — [{"start": int, "length": int}, ...] per example
    """

    def __init__(self, data_dir: Path, split: str, max_seq_len: int, pad_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # Load binary data as memory-mapped arrays
        ids_path = data_dir / f"{split}.bin"
        mask_path = data_dir / f"{split}.mask.bin"
        offsets_path = data_dir / f"{split}.offsets.json"

        self.ids = np.memmap(str(ids_path), dtype=np.uint32, mode="r")
        self.masks = np.memmap(str(mask_path), dtype=np.uint8, mode="r")

        with open(offsets_path, "r") as f:
            self.offsets = json.load(f)

        self.n_examples = len(self.offsets)

        # Precompute stats
        self.total_tokens = sum(o["length"] for o in self.offsets)
        lengths = [o["length"] for o in self.offsets]
        self.avg_len = self.total_tokens / self.n_examples if self.n_examples else 0
        self.max_len = max(lengths) if lengths else 0
        self.min_len = min(lengths) if lengths else 0

    def __len__(self):
        return self.n_examples

    def get_example(self, idx: int):
        """Get a single example (token IDs and loss mask)."""
        offset = self.offsets[idx]
        start = offset["start"]
        length = min(offset["length"], self.max_seq_len)

        ids = self.ids[start:start + length].astype(np.int64)
        mask = self.masks[start:start + length].astype(np.int64)

        return ids, mask

    def get_batch(self, indices: list[int], device: str = "cuda"):
        """Get a padded batch of examples.

        Returns:
            input_ids: (B, T) — token IDs, right-padded with pad_id
            targets:   (B, T) — shifted targets, -100 where mask=0 or padding
            loss_mask: (B, T) — 1 where loss should be computed
        """
        import torch

        batch_ids = []
        batch_masks = []

        for idx in indices:
            ids, mask = self.get_example(idx)
            batch_ids.append(ids)
            batch_masks.append(mask)

        # Find max length in this batch for padding
        max_len = max(len(ids) for ids in batch_ids)

        # Pad to max_len
        padded_ids = np.full((len(indices), max_len), self.pad_id, dtype=np.int64)
        padded_masks = np.zeros((len(indices), max_len), dtype=np.int64)

        for i, (ids, mask) in enumerate(zip(batch_ids, batch_masks)):
            padded_ids[i, :len(ids)] = ids
            padded_masks[i, :len(ids)] = mask

        # Build input/target pairs
        # Input: tokens[:-1], Target: tokens[1:]
        # The model predicts next token, so shift by 1
        input_ids = torch.from_numpy(padded_ids[:, :-1]).to(device)
        target_ids = torch.from_numpy(padded_ids[:, 1:]).to(device)
        target_mask = torch.from_numpy(padded_masks[:, 1:]).to(device)

        # Set targets to -100 where we don't want loss
        # -100 is ignored by F.cross_entropy(ignore_index=-100)
        targets = target_ids.clone()
        targets[target_mask == 0] = -100

        return input_ids, targets


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
    """Segment raw text using Morfessor with @@ continuation markers."""
    tokens = text.split()
    seg_tokens = []

    for token in tokens:
        if TELUGU_WORD_RE.fullmatch(token):
            segments = morf_model.viterbi_segment(token)[0]
            for i, seg in enumerate(segments):
                if i < len(segments) - 1:
                    seg_tokens.append(seg + separator)
                else:
                    seg_tokens.append(seg)
        elif TELUGU_WORD_RE.search(token):
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
            seg_tokens.append(token)

    return " ".join(seg_tokens)


# ============================================================================
# Model Loading & Embedding Resize
# ============================================================================
def load_pretrained_and_resize(checkpoint_path: Path, new_vocab_size: int, device: str = "cuda"):
    """Load pretrained checkpoint and resize embedding/lm_head for new special tokens.

    The pretrained model has vocab_size = V (e.g. 42000).
    The SFT model needs vocab_size = V + 4 (for <|system|>, <|user|>, <|assistant|>, <|end|>).

    Strategy:
      1. Load checkpoint config, set vocab_size = new_vocab_size
      2. Build model with new vocab_size
      3. Load pretrained weights into the model (partial — old embedding rows)
      4. Initialize new embedding rows to mean of existing embeddings
      5. Weight tying (wte = lm_head) is maintained automatically
    """
    import torch

    sys.path.insert(0, str(Path(__file__).parent))
    from train_gpt import GPTConfig, build_model

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    old_config = GPTConfig(**checkpoint["config"])
    old_vocab_size = old_config.vocab_size

    logger.info("Loaded pretrained checkpoint from %s", checkpoint_path)
    logger.info("  Old vocab size: %d", old_vocab_size)
    logger.info("  New vocab size: %d (+%d special tokens)", new_vocab_size, new_vocab_size - old_vocab_size)
    logger.info("  Step: %s, Val loss: %s",
                checkpoint.get("step", "?"), checkpoint.get("val_loss", "?"))

    # Build model with new vocab size
    new_config = GPTConfig(
        block_size=old_config.block_size,
        vocab_size=new_vocab_size,
        n_layer=old_config.n_layer,
        n_head=old_config.n_head,
        n_embd=old_config.n_embd,
        dropout=old_config.dropout,
        bias=old_config.bias,
        rope_theta=old_config.rope_theta,
    )

    model = build_model(new_config, device)

    # Load pretrained weights — need to handle embedding size mismatch
    old_state = checkpoint["model"]
    new_state = model.state_dict()

    # Copy all weights except embedding (which has different size)
    for key in old_state:
        if key in new_state:
            if old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]
            elif "wte.weight" in key or "lm_head.weight" in key:
                # Embedding/lm_head: copy old rows, init new rows
                old_weight = old_state[key]
                new_state[key][:old_vocab_size] = old_weight
                # Init new rows to mean of existing embeddings
                mean_emb = old_weight.mean(dim=0)
                for i in range(old_vocab_size, new_vocab_size):
                    new_state[key][i] = mean_emb
                logger.info("  Resized %s: %s -> %s (new rows init to mean)",
                            key, list(old_weight.shape), list(new_state[key].shape))
            else:
                logger.warning("  Shape mismatch for %s: %s vs %s — skipping",
                               key, list(old_state[key].shape), list(new_state[key].shape))

    model.load_state_dict(new_state)
    model.to(device)

    n_params = model.count_parameters()
    logger.info("  Model params: %d (%.1fM)", n_params, n_params / 1e6)

    return model, new_config


# ============================================================================
# Checkpoint Saving
# ============================================================================
def save_sft_checkpoint(model, optimizer, config, path, **kwargs):
    """Save SFT checkpoint with full metadata."""
    import torch
    state = model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict()
    ckpt = {
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config.__dict__,
        "architecture": "llama",
        "training_type": "sft",
    }
    ckpt.update(kwargs)
    torch.save(ckpt, str(path))


# ============================================================================
# Training Loop
# ============================================================================
def train_sft(
    checkpoint_path: Path,
    data_dir: Path,
    tokenizer_dir: Path,
    morfessor_path: Path,
    sft_config: SFTConfig,
    resume_from: str = None,
):
    """Main SFT training loop."""
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if sft_config.dtype == "bfloat16" else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()

    logger.info("=" * 70)
    logger.info("Telugu LLaMA — Supervised Fine-Tuning (Full)")
    logger.info("=" * 70)

    # ---- Load SFT dataset metadata ----
    meta_path = data_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    new_vocab_size = meta["vocab_size_with_special"]
    special_tokens = meta["special_tokens"]
    max_seq_len = meta["max_seq_len"]
    pad_id = special_tokens.get("<pad>", 0)

    logger.info("  SFT data dir:    %s", data_dir)
    logger.info("  New vocab size:  %d", new_vocab_size)
    logger.info("  Max seq len:     %d", max_seq_len)
    logger.info("  Special tokens:  %s", special_tokens)

    # ---- Load datasets ----
    train_data = SFTDataset(data_dir, "train", max_seq_len, pad_id)
    val_data = SFTDataset(data_dir, "val", max_seq_len, pad_id)

    logger.info("  Train: %d examples, %d tokens, avg_len=%.0f, max_len=%d",
                train_data.n_examples, train_data.total_tokens,
                train_data.avg_len, train_data.max_len)
    logger.info("  Val:   %d examples, %d tokens, avg_len=%.0f, max_len=%d",
                val_data.n_examples, val_data.total_tokens,
                val_data.avg_len, val_data.max_len)

    # ---- Load pretrained model and resize ----
    if resume_from and os.path.exists(resume_from):
        logger.info("Resuming SFT from %s", resume_from)
        resume_ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        from train_gpt import GPTConfig, build_model
        model_config = GPTConfig(**resume_ckpt["config"])
        model = build_model(model_config, device)
        model.load_state_dict(resume_ckpt["model"])
        model.to(device)
        start_step = resume_ckpt.get("step", 0)
        start_epoch = resume_ckpt.get("epoch", 0)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        logger.info("  Resumed at step %d, epoch %d, best_val=%.4f",
                     start_step, start_epoch, best_val_loss)
    else:
        model, model_config = load_pretrained_and_resize(
            checkpoint_path, new_vocab_size, device
        )
        start_step = 0
        start_epoch = 0
        best_val_loss = float("inf")

    # ---- Training plan ----
    steps_per_epoch = math.ceil(
        train_data.n_examples / (sft_config.micro_batch_size * sft_config.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * sft_config.max_epochs

    tokens_per_step = sft_config.micro_batch_size * sft_config.gradient_accumulation_steps * int(train_data.avg_len)

    logger.info("  Micro batch:     %d", sft_config.micro_batch_size)
    logger.info("  Grad accum:      %d", sft_config.gradient_accumulation_steps)
    logger.info("  Effective batch: %d", sft_config.micro_batch_size * sft_config.gradient_accumulation_steps)
    logger.info("  Steps/epoch:     %d", steps_per_epoch)
    logger.info("  Total steps:     %d (%d epochs)", total_steps, sft_config.max_epochs)
    logger.info("  Learning rate:   %.2e -> %.2e", sft_config.learning_rate, sft_config.min_lr)
    logger.info("  Weight decay:    %.4f", sft_config.weight_decay)

    # ---- Compile ----
    if sft_config.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ---- Optimizer ----
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": sft_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=sft_config.learning_rate,
        betas=(sft_config.beta1, sft_config.beta2),
        fused=True if device == "cuda" else False,
    )

    if resume_from and os.path.exists(resume_from):
        if resume_ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            logger.info("Restored optimizer state")
        del resume_ckpt

    # ---- LR schedule (cosine with warmup) ----
    def get_lr(step):
        if step < sft_config.warmup_steps:
            return sft_config.learning_rate * (step + 1) / sft_config.warmup_steps
        if step >= total_steps:
            return sft_config.min_lr
        decay_ratio = (step - sft_config.warmup_steps) / (total_steps - sft_config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return sft_config.min_lr + coeff * (sft_config.learning_rate - sft_config.min_lr)

    # ---- Save dir ----
    save_dir = Path(sft_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer + Morfessor for sample generation ----
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer

    tokenizer = MorfessorTokenizer(tokenizer_dir)
    morf_model = load_morfessor_model(morfessor_path) if morfessor_path.exists() else None

    # Build reverse lookup for special tokens
    id_to_special = {v: k for k, v in special_tokens.items()}

    # ---- Sample prompts for eval generation ----
    # Pick a few val examples and use just the prompt portion (mask=0 tokens)
    sample_prompts = []
    NUM_SAMPLES = min(3, val_data.n_examples)
    for i in range(NUM_SAMPLES):
        ids, mask = val_data.get_example(i)
        # Prompt = all tokens before first mask=1 token
        prompt_end = 0
        for j, m in enumerate(mask):
            if m == 1:
                prompt_end = j
                break
        if prompt_end > 0:
            sample_prompts.append(ids[:prompt_end].tolist())

    # ---- W&B ----
    use_wandb = bool(sft_config.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb_config = {
                **sft_config.__dict__,
                "model_config": model_config.__dict__,
                "new_vocab_size": new_vocab_size,
                "train_examples": train_data.n_examples,
                "val_examples": val_data.n_examples,
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "training_type": "sft_full",
            }
            wandb.init(
                project=sft_config.wandb_project,
                name=sft_config.wandb_run_name or None,
                config=wandb_config,
                resume="allow" if resume_from else None,
            )
            logger.info("W&B logging enabled: project=%s", sft_config.wandb_project)
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            use_wandb = False

    # ---- Training loop ----
    logger.info("=" * 70)
    logger.info("Starting SFT training...")
    logger.info("=" * 70)

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    model.train()

    rng = np.random.RandomState(42)
    t0 = time.time()
    global_step = start_step
    tokens_processed = 0
    no_improve_count = 0
    patience = 3  # stop if val loss doesn't improve for this many evals

    for epoch in range(start_epoch, sft_config.max_epochs):
        # Shuffle examples each epoch
        indices = rng.permutation(train_data.n_examples).tolist()

        # Process in batches
        effective_batch = sft_config.micro_batch_size * sft_config.gradient_accumulation_steps
        n_batches = math.ceil(len(indices) / effective_batch)

        for batch_idx in range(n_batches):
            # LR schedule
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            tokens_in_step = 0

            for micro_step in range(sft_config.gradient_accumulation_steps):
                # Get micro-batch indices
                start_idx = batch_idx * effective_batch + micro_step * sft_config.micro_batch_size
                end_idx = min(start_idx + sft_config.micro_batch_size, len(indices))
                if start_idx >= len(indices):
                    break
                batch_indices = indices[start_idx:end_idx]
                if not batch_indices:
                    break

                input_ids, targets = train_data.get_batch(batch_indices, device)
                tokens_in_step += input_ids.numel()

                with ctx:
                    logits, _ = model(input_ids)
                    # Compute masked loss manually (model's built-in uses ignore_index=-1)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                    )
                    loss = loss / sft_config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                loss_accum += loss.item()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            tokens_processed += tokens_in_step
            global_step += 1

            # ---- Logging ----
            if global_step % sft_config.log_interval == 0:
                dt = time.time() - t0
                tokens_per_sec = tokens_processed / dt if dt > 0 else 0
                logger.info(
                    "step %4d/%d | epoch %d | loss %.4f | lr %.2e | %.0f tok/s | %.1f min",
                    global_step, total_steps, epoch + 1, loss_accum, lr,
                    tokens_per_sec, dt / 60,
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": loss_accum,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                        "train/tokens_per_sec": tokens_per_sec,
                    }, step=global_step)

            # ---- Evaluation ----
            if global_step % sft_config.eval_interval == 0:
                model.eval()
                val_losses = []
                val_tokens = 0

                # Eval on full val set (or eval_steps batches)
                val_indices = list(range(val_data.n_examples))
                eval_batch_size = sft_config.micro_batch_size
                n_eval_batches = math.ceil(len(val_indices) / eval_batch_size)

                if sft_config.eval_steps > 0:
                    n_eval_batches = min(n_eval_batches, sft_config.eval_steps)

                with torch.no_grad():
                    for eb in range(n_eval_batches):
                        eb_start = eb * eval_batch_size
                        eb_end = min(eb_start + eval_batch_size, len(val_indices))
                        eb_indices = val_indices[eb_start:eb_end]

                        input_ids, targets = val_data.get_batch(eb_indices, device)
                        n_loss_tokens = (targets != -100).sum().item()

                        with ctx:
                            logits, _ = model(input_ids)
                            loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                targets.view(-1),
                                ignore_index=-100,
                                reduction="sum",
                            )
                        val_losses.append(loss.item())
                        val_tokens += n_loss_tokens

                val_loss = sum(val_losses) / val_tokens if val_tokens > 0 else float("inf")
                logger.info("step %4d | val_loss %.4f (on %d tokens)", global_step, val_loss, val_tokens)

                # ---- Sample generation ----
                samples = []
                if sample_prompts:
                    with torch.no_grad():
                        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        for prompt_ids in sample_prompts:
                            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                            y = raw_model.generate(x, max_new_tokens=80, temperature=0.7, top_k=40)
                            gen_ids = y[0].tolist()
                            # Decode with special token handling
                            prompt_text = _decode_with_specials(prompt_ids, tokenizer, id_to_special)
                            gen_text = _decode_with_specials(gen_ids[len(prompt_ids):], tokenizer, id_to_special)
                            samples.append({"prompt": prompt_text, "generated": gen_text})
                            logger.info("  [sample] %s → %s",
                                        prompt_text[:80], gen_text[:100])

                model.train()

                if use_wandb:
                    wandb.log({"val/loss": val_loss}, step=global_step)
                    if samples:
                        table = wandb.Table(columns=["step", "prompt", "generated"])
                        for s in samples:
                            table.add_data(global_step, s["prompt"], s["generated"])
                        wandb.log({"val/samples": table}, step=global_step)

                # ---- Best model & early stopping ----
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    ckpt_path = save_dir / "best.pt"
                    save_sft_checkpoint(model, optimizer, model_config, ckpt_path,
                                        step=global_step, epoch=epoch,
                                        val_loss=val_loss, best_val_loss=best_val_loss,
                                        special_tokens=special_tokens)
                    logger.info("New best! val_loss=%.4f. Saved to %s", val_loss, ckpt_path)
                else:
                    no_improve_count += 1
                    logger.info("No improvement (%d/%d patience)", no_improve_count, patience)
                    if no_improve_count >= patience:
                        logger.info("Early stopping triggered after %d evals without improvement.", patience)
                        break

            # ---- Periodic save ----
            if global_step % sft_config.save_interval == 0:
                ckpt_path = save_dir / f"step_{global_step:05d}.pt"
                save_sft_checkpoint(model, optimizer, model_config, ckpt_path,
                                    step=global_step, epoch=epoch,
                                    val_loss=val_loss if "val_loss" in dir() else None,
                                    best_val_loss=best_val_loss,
                                    special_tokens=special_tokens)
                logger.info("Checkpoint saved to %s", ckpt_path)

        # Check if early stopping was triggered
        if no_improve_count >= patience:
            break

        # Epoch complete
        logger.info("Epoch %d complete (step %d)", epoch + 1, global_step)

    # ---- Final save ----
    total_time = time.time() - t0
    ckpt_path = save_dir / "final.pt"
    save_sft_checkpoint(model, optimizer, model_config, ckpt_path,
                        step=global_step, epoch=epoch,
                        best_val_loss=best_val_loss,
                        special_tokens=special_tokens)

    logger.info("=" * 70)
    logger.info("SFT Training complete!")
    logger.info("  Total steps:     %d", global_step)
    logger.info("  Total time:      %.1f minutes", total_time / 60)
    logger.info("  Best val loss:   %.4f", best_val_loss)
    logger.info("  Tokens processed: %d", tokens_processed)
    logger.info("  Final checkpoint: %s", ckpt_path)
    logger.info("  Best checkpoint:  %s", save_dir / "best.pt")
    logger.info("=" * 70)

    if use_wandb:
        wandb.log({
            "final/best_val_loss": best_val_loss,
            "final/total_minutes": total_time / 60,
        }, step=global_step)
        wandb.finish()


# ============================================================================
# Decode helper
# ============================================================================
def _decode_with_specials(token_ids: list[int], tokenizer, id_to_special: dict) -> str:
    """Decode token IDs to text, handling special tokens."""
    parts = []
    regular_ids = []

    for tid in token_ids:
        special_name = id_to_special.get(tid)
        if special_name:
            # Flush regular tokens
            if regular_ids:
                parts.append(tokenizer.decode(regular_ids))
                regular_ids = []
            parts.append(special_name)
        else:
            regular_ids.append(tid)

    # Flush remaining
    if regular_ids:
        parts.append(tokenizer.decode(regular_ids))

    return " ".join(parts)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Telugu LLaMA — Supervised Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", "-c", type=Path, required=True,
        help="Path to pretrained checkpoint (best.pt from pretraining)",
    )
    parser.add_argument(
        "--data", "-d", type=Path, required=True,
        help="Path to SFT data directory (from build_sft_dataset.py)",
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
        "--save-dir", type=str, default="./sft_checkpoints",
        help="Checkpoint output directory (default: ./sft_checkpoints)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Peak learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Micro batch size (default: 8)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=50,
        help="Steps between evaluations (default: 50)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="Steps between checkpoints (default: 100)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience — evals without improvement (default: 3)",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume SFT from a previous SFT checkpoint",
    )
    parser.add_argument(
        "--wandb", type=str, default="",
        help="W&B project name (enables logging). E.g. --wandb pothana-sft",
    )
    parser.add_argument(
        "--wandb-name", type=str, default="",
        help="W&B run name (optional)",
    )

    args = parser.parse_args()

    sft_config = SFTConfig()
    sft_config.max_epochs = args.epochs
    sft_config.learning_rate = args.lr
    sft_config.min_lr = args.lr / 10
    sft_config.micro_batch_size = args.batch_size
    sft_config.gradient_accumulation_steps = args.grad_accum
    sft_config.eval_interval = args.eval_interval
    sft_config.save_interval = args.save_interval
    sft_config.save_dir = args.save_dir
    sft_config.compile_model = not args.no_compile
    sft_config.wandb_project = args.wandb
    sft_config.wandb_run_name = args.wandb_name

    train_sft(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        tokenizer_dir=args.tokenizer,
        morfessor_path=args.morfessor,
        sft_config=sft_config,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
