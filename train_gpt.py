#!/usr/bin/env python3
"""
Telugu LLaMA Training Script
==============================
Trains a ~300M parameter LLaMA-style model on Morfessor-segmented Telugu text.

Architecture (LLaMA-style):
  - RMSNorm (instead of LayerNorm)
  - Rotary Positional Embeddings / RoPE (instead of learned absolute)
  - SwiGLU MLP (instead of GELU MLP)
  - Pre-norm transformer decoder, weight-tied embeddings

Training: 3 epochs (~45K steps) with aggressive checkpointing.

Two-phase operation:
  1. prepare  — tokenize segmented corpus into memory-mapped binary shards
  2. train    — train model on the prepared shards

Optimized for single A100 80GB with:
  - Flash Attention via PyTorch SDPA
  - bf16 mixed precision
  - Gradient checkpointing (optional)
  - Memory-mapped data loading (no RAM bottleneck)
  - Cosine LR schedule with warmup

Usage:
    # Step 1: Prepare data (tokenize segmented corpus into binary shards)
    python train_gpt.py prepare --data ./data/morfessor/segmented_corpus --tokenizer ./tokenizer

    # Step 2: Train (3 epochs with W&B logging)
    python train_gpt.py train --data ./train_data --tokenizer ./tokenizer --wandb telugu-gpt

    # Or do both in one go
    python train_gpt.py all --data ./data/morfessor/segmented_corpus --tokenizer ./tokenizer
"""

import os
import sys
import math
import json
import time
import struct
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Model Configuration
# ===========================================================================
@dataclass
class GPTConfig:
    """LLaMA-style model configuration — ~300M parameters."""
    block_size: int = 2048       # context length
    vocab_size: int = 0          # 0 = auto-detect from tokenizer at runtime
    n_layer: int = 20            # transformer layers
    n_head: int = 16             # attention heads
    n_embd: int = 1024           # embedding dimension
    dropout: float = 0.1
    bias: bool = False           # no bias in linear layers
    rope_theta: float = 10000.0  # RoPE base frequency

    def param_count(self):
        """Rough parameter count estimate."""
        # Embedding: vocab * embd (no positional embedding — using RoPE)
        emb = self.vocab_size * self.n_embd
        # Attention per layer: QKV (3 * n_embd^2) + output proj (n_embd^2) = 4 * n_embd^2
        attn_per_layer = 4 * self.n_embd ** 2
        # SwiGLU MLP per layer: gate + up + down = 3 * n_embd * hidden_dim
        hidden_dim = ((int(2 * self.n_embd * 4 / 3) + 255) // 256) * 256
        mlp_per_layer = 3 * self.n_embd * hidden_dim
        tfm = self.n_layer * (attn_per_layer + mlp_per_layer)
        # RMSNorm params: (2 per layer + 1 final) * n_embd
        norms = (2 * self.n_layer + 1) * self.n_embd
        # LM head shares weights with embedding
        return emb + tfm + norms


# ===========================================================================
# Training Configuration
# ===========================================================================
@dataclass
class TrainConfig:
    """Training hyperparameters optimized for single A100 80GB."""
    # Batch
    micro_batch_size: int = 32          # sequences per GPU step
    gradient_accumulation_steps: int = 4 # effective batch = 32 * 4 = 128 sequences
    # effective tokens per step = 128 * 2048 = 262,144

    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 3e-5                 # 10% of max LR
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 500
    max_steps: int = 45000               # ~3 epochs over 3.7B tokens
    lr_decay_steps: int = 45000          # cosine decay over full training

    # Logging & saving
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 20                 # batches for eval loss
    save_interval: int = 500             # aggressive checkpointing (every 500 steps)
    save_dir: str = "./checkpoints"

    # Data
    val_split: float = 0.02              # 2% for validation

    # Performance
    compile_model: bool = True           # torch.compile
    dtype: str = "bfloat16"              # bf16 on A100
    gradient_checkpointing: bool = False # set True if OOM

    # Logging
    wandb_project: str = ""              # set to enable W&B logging (e.g. "telugu-gpt")
    wandb_run_name: str = ""             # optional run name (auto-generated if empty)


# ===========================================================================
# Phase 1: Data Preparation
# ===========================================================================
def prepare_data(
    data_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    val_split: float,
    block_size: int,
    num_workers: int = 0,
):
    """
    Tokenize segmented corpus into memory-mapped binary shards.

    Simple single-threaded approach with per-line progress bar and
    streaming writes to disk. Uses encode_lines_to_array() with BPE
    cache for speed.

    Reads .seg.txt files, encodes with our Morfessor tokenizer,
    and writes train.bin / val.bin as uint32 numpy arrays.
    """
    from tqdm import tqdm

    # Import our tokenizer
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer

    tokenizer = MorfessorTokenizer(tokenizer_dir)
    logger.info("Loaded tokenizer: vocab_size=%d", tokenizer.vocab_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already prepared
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    if train_bin.exists() and val_bin.exists():
        train_tokens = os.path.getsize(train_bin) // 4  # uint32
        val_tokens = os.path.getsize(val_bin) // 4
        logger.info("Data already prepared:")
        logger.info("  train.bin: %d tokens (%.2f GB)", train_tokens, os.path.getsize(train_bin) / 1e9)
        logger.info("  val.bin:   %d tokens (%.2f GB)", val_tokens, os.path.getsize(val_bin) / 1e9)
        logger.info("  Delete these files to re-prepare.")
        return train_tokens, val_tokens

    # Find all segmented files
    seg_files = sorted(data_dir.rglob("*.seg.txt"))
    if not seg_files:
        seg_files = sorted(data_dir.rglob("*.txt"))
        seg_files = [f for f in seg_files if "morfessor" not in str(f)]

    if not seg_files:
        logger.error("No segmented text files found in %s", data_dir)
        sys.exit(1)

    logger.info("Found %d files to tokenize", len(seg_files))

    # Local refs for hot-path speed
    _get = tokenizer.token_to_id.get
    _unk = tokenizer.unk_id
    _eos = tokenizer.eos_id
    _sep = tokenizer.separator
    _sep_len = len(_sep)
    _is_tel = tokenizer._is_telugu
    _bpe = tokenizer._encode_token_bpe
    _id2tok = tokenizer.id_to_token

    # Stream-write to temp binary (avoids holding all IDs in RAM)
    temp_bin = output_dir / "all_tokens.tmp.bin"
    total_tokens = 0
    total_unk = 0

    # Micro-batch: encode N lines at a time, write as one array
    BATCH_SIZE = 1000  # lines per micro-batch write

    with open(temp_bin, "wb") as out_f:
        for fpath in seg_files:
            fsize_mb = os.path.getsize(fpath) / 1e6
            logger.info("  Tokenizing %s (%.0f MB)...", fpath.name, fsize_mb)

            batch_ids = []
            batch_unk = 0
            batch_total = 0
            line_in_batch = 0

            with open(fpath, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=fpath.name, unit=" lines", mininterval=0.5):
                    line = line.strip()
                    if not line:
                        continue

                    # Inline encode — avoids method call overhead per line
                    for token in line.split():
                        tid = _get(token)
                        if tid is not None:
                            batch_ids.append(tid)
                            batch_total += 1
                            if tid == _unk:
                                batch_unk += 1
                            continue

                        # Slow path: fallback
                        batch_total += 1
                        is_cont = token.endswith(_sep)
                        bare = token[:-_sep_len] if is_cont else token

                        if not _is_tel(bare):
                            sub_ids = _bpe(bare)
                            if is_cont and sub_ids:
                                last_str = _id2tok.get(sub_ids[-1], "")
                                if not last_str.endswith(_sep):
                                    ct = _get(last_str + _sep)
                                    if ct is not None:
                                        sub_ids[-1] = ct
                            batch_ids.extend(sub_ids)
                            batch_unk += sum(1 for i in sub_ids if i == _unk)
                        else:
                            n = len(bare)
                            if is_cont:
                                for i, ch in enumerate(bare):
                                    if i < n - 1:
                                        batch_ids.append(_get(ch + _sep, _unk))
                                    else:
                                        batch_ids.append(_get(ch + _sep, _get(ch, _unk)))
                            else:
                                for i, ch in enumerate(bare):
                                    if i < n - 1:
                                        batch_ids.append(_get(ch + _sep, _unk))
                                    else:
                                        batch_ids.append(_get(ch, _unk))

                    # Append EOS after each document
                    batch_ids.append(_eos)
                    batch_total += 1
                    line_in_batch += 1

                    # Flush micro-batch to disk periodically
                    if line_in_batch >= BATCH_SIZE:
                        arr = np.array(batch_ids, dtype=np.uint32)
                        out_f.write(arr.tobytes())
                        total_tokens += batch_total
                        total_unk += batch_unk
                        batch_ids = []
                        batch_unk = 0
                        batch_total = 0
                        line_in_batch = 0

            # Flush remaining
            if batch_ids:
                arr = np.array(batch_ids, dtype=np.uint32)
                out_f.write(arr.tobytes())
                total_tokens += batch_total
                total_unk += batch_unk

    logger.info("Total tokens: %d", total_tokens)
    logger.info("UNK tokens:   %d (%.2f%%)", total_unk, 100 * total_unk / total_tokens if total_tokens else 0)

    # Phase 2: Split into train/val from the temp binary
    temp_size = os.path.getsize(temp_bin)
    n_total = temp_size // 4  # uint32 = 4 bytes
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    logger.info("Splitting: %d train + %d val tokens", n_train, n_val)

    all_data = np.memmap(str(temp_bin), dtype=np.uint32, mode="r", shape=(n_total,))
    all_data[:n_train].tofile(str(train_bin))
    all_data[n_train:].tofile(str(val_bin))
    del all_data
    temp_bin.unlink()

    train_gb = os.path.getsize(train_bin) / 1e9
    val_gb = os.path.getsize(val_bin) / 1e9

    logger.info("Saved prepared data:")
    logger.info("  train.bin: %d tokens (%.2f GB)", n_train, train_gb)
    logger.info("  val.bin:   %d tokens (%.2f GB)", n_val, val_gb)
    logger.info("  Location:  %s", output_dir.resolve())

    # Save metadata
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "total_tokens": int(total_tokens),
        "unk_rate": total_unk / total_tokens if total_tokens else 0,
        "dtype": "uint32",
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return n_train, n_val


# ===========================================================================
# GPT Model
# ===========================================================================
def build_model(config: GPTConfig, device: str = "cuda"):
    """Build LLaMA-style model from config (RoPE + SwiGLU + RMSNorm)."""
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    # ----- RMSNorm (replaces LayerNorm) -----
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    # ----- RoPE helpers -----
    def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
        """Precompute complex-valued rotation frequencies for RoPE."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)          # (max_seq_len, dim//2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def apply_rotary_emb(xq, xk, freqs_cis):
        """Apply rotary embeddings to Q and K tensors."""
        # xq, xk: (B, n_head, T, head_dim)
        # freqs_cis: (T, head_dim//2) complex
        B, H, T, D = xq.shape
        xq_ = xq.float().reshape(B, H, T, D // 2, 2)
        xk_ = xk.float().reshape(B, H, T, D // 2, 2)
        xq_complex = torch.view_as_complex(xq_)
        xk_complex = torch.view_as_complex(xk_)
        freqs = freqs_cis.unsqueeze(0).unsqueeze(0)   # (1, 1, T, D//2)
        xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    # ----- Attention with RoPE -----
    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.resid_dropout = nn.Dropout(config.dropout)
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.head_dim = config.n_embd // config.n_head
            self.dropout = config.dropout

        def forward(self, x, freqs_cis):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            # Apply RoPE to Q and K (not V)
            q, k = apply_rotary_emb(q, k, freqs_cis)
            # Flash attention (PyTorch >= 2.0)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y

    # ----- SwiGLU MLP (replaces GELU MLP) -----
    class SwiGLUMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            # LLaMA convention: hidden = round_up(4 * n_embd * 2/3, 256)
            hidden_dim = int(2 * config.n_embd * 4 / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
            self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))

    # ----- Transformer Block -----
    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = RMSNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = RMSNorm(config.n_embd)
            self.mlp = SwiGLUMLP(config)

        def forward(self, x, freqs_cis):
            x = x + self.attn(self.ln_1(x), freqs_cis)
            x = x + self.mlp(self.ln_2(x))
            return x

    # ----- GPT (LLaMA-style) -----
    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # Weight tying
            self.transformer.wte.weight = self.lm_head.weight

            # Precompute RoPE frequencies and store as buffer
            head_dim = config.n_embd // config.n_head
            freqs_cis = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
            self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis))

            # Init weights
            self.apply(self._init_weights)
            # Scale residual projections (attention c_proj + SwiGLU w_down)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                pass  # weight already initialized to ones in constructor

        def forward(self, idx, targets=None):
            device = idx.device
            B, T = idx.size()
            assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)

            # Recover complex freqs from stored real buffer, slice to seq length
            freqs_cis = torch.view_as_complex(self.freqs_cis[:T])

            for block in self.transformer.h:
                x = block(x, freqs_cis)
            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None

            return logits, loss

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters())

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    model = GPT(config)
    return model


# ===========================================================================
# Dataset
# ===========================================================================
class MemmapDataset:
    """Memory-mapped dataset for pretraining. Zero RAM overhead."""

    def __init__(self, data_path: Path, block_size: int):
        self.data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
        self.block_size = block_size
        self.n_tokens = len(self.data)

    def __len__(self):
        return self.n_tokens // self.block_size

    def get_batch(self, batch_size: int, device: str = "cuda"):
        import torch
        ix = np.random.randint(0, self.n_tokens - self.block_size - 1, (batch_size,))
        x = np.stack([self.data[i:i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y


# ===========================================================================
# Checkpoint Helper
# ===========================================================================
def _save_checkpoint(model, optimizer, model_config, path, **kwargs):
    """Save a checkpoint with full metadata. Never loses data."""
    import torch
    state = model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict()
    ckpt = {
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": model_config.__dict__,
        "architecture": "llama",
    }
    ckpt.update(kwargs)
    torch.save(ckpt, str(path))


# ===========================================================================
# Training Loop
# ===========================================================================
def train(
    data_dir: Path,
    tokenizer_dir: Path,
    model_config: GPTConfig,
    train_config: TrainConfig,
    resume_from: str = None,
):
    """Main training loop."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if train_config.dtype == "bfloat16" else torch.float16
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()

    logger.info("=" * 70)
    logger.info("Telugu LLaMA Training (RoPE + SwiGLU + RMSNorm)")
    logger.info("=" * 70)

    # Load data
    train_data = MemmapDataset(data_dir / "train.bin", model_config.block_size)
    val_data = MemmapDataset(data_dir / "val.bin", model_config.block_size)

    logger.info("  Train tokens: %d (%.2f GB)", train_data.n_tokens, train_data.n_tokens * 4 / 1e9)
    logger.info("  Val tokens:   %d", val_data.n_tokens)

    # Read vocab size from metadata (saved during prepare step)
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_config.vocab_size = meta["vocab_size"]
        logger.info("  Vocab size:   %d (from meta.json)", model_config.vocab_size)
    elif model_config.vocab_size <= 0:
        logger.error("meta.json not found in %s and no vocab_size set.", data_dir)
        logger.error("Run the 'prepare' step first.")
        sys.exit(1)

    # Build model
    model = build_model(model_config, device)
    n_params = model.count_parameters()
    logger.info("  Model params: %d (%.1fM)", n_params, n_params / 1e6)
    logger.info("  Block size:   %d", model_config.block_size)
    logger.info("  Layers:       %d", model_config.n_layer)
    logger.info("  Heads:        %d", model_config.n_head)
    logger.info("  Embed dim:    %d", model_config.n_embd)

    # Effective batch
    tokens_per_step = (
        train_config.micro_batch_size
        * train_config.gradient_accumulation_steps
        * model_config.block_size
    )
    total_steps = train_config.max_steps
    total_tokens_trained = tokens_per_step * total_steps

    logger.info("  Micro batch:  %d", train_config.micro_batch_size)
    logger.info("  Grad accum:   %d", train_config.gradient_accumulation_steps)
    logger.info("  Tokens/step:  %d", tokens_per_step)
    logger.info("  Total steps:  %d", total_steps)
    logger.info("  Total tokens: %d (%.1fx data)", total_tokens_trained, total_tokens_trained / train_data.n_tokens)

    # Epoch tracking
    tokens_per_epoch = train_data.n_tokens
    steps_per_epoch = tokens_per_epoch / tokens_per_step
    n_epochs = total_tokens_trained / tokens_per_epoch

    logger.info("  Tokens/epoch: %d (%.2f GB)", tokens_per_epoch, tokens_per_epoch * 4 / 1e9)
    logger.info("  Steps/epoch:  %.0f", steps_per_epoch)
    logger.info("  Epochs:       %.2f", n_epochs)

    # Estimated time
    est_tokens_per_sec = 150_000  # conservative A100 estimate
    est_hours = total_tokens_trained / est_tokens_per_sec / 3600
    logger.info("  Est. time:    %.1f hours (at ~%dK tok/s)", est_hours, est_tokens_per_sec // 1000)

    # Checkpoint disk estimate
    n_checkpoints = total_steps // train_config.save_interval + int(n_epochs) + 1
    est_ckpt_size_gb = n_params * 4 * 2 / 1e9  # model + optimizer, fp32
    logger.info("  Checkpoints:  ~%d saves, ~%.0f GB total disk", n_checkpoints, n_checkpoints * est_ckpt_size_gb)
    logger.info("=" * 70)

    model.to(device)

    # Compile
    if train_config.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Gradient checkpointing
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=True if device == "cuda" else False,
    )

    # LR schedule
    def get_lr(step):
        if step < train_config.warmup_steps:
            return train_config.learning_rate * step / train_config.warmup_steps
        if step > train_config.lr_decay_steps:
            return train_config.min_lr
        decay_ratio = (step - train_config.warmup_steps) / (train_config.lr_decay_steps - train_config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

    # Resume
    start_step = 0
    best_val_loss = float("inf")
    tokens_processed = 0
    if resume_from and os.path.exists(resume_from):
        logger.info("Resuming from %s", resume_from)
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        tokens_processed = checkpoint.get("tokens_processed", start_step * tokens_per_step)
        logger.info("Resumed at step %d (epoch %.2f), best_val_loss=%.4f",
                     start_step, tokens_processed / tokens_per_epoch, best_val_loss)

    # Save dir
    save_dir = Path(train_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for sample generation during eval
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer
    tokenizer = MorfessorTokenizer(tokenizer_dir)

    # Sample prompts for generation during eval (seeded from val data)
    sample_prompts_ids = []
    SAMPLE_PROMPT_LEN = 32   # tokens of context to feed as prompt
    SAMPLE_GEN_LEN = 64      # tokens to generate
    NUM_SAMPLES = 3
    for _ in range(NUM_SAMPLES):
        ix = np.random.randint(0, val_data.n_tokens - SAMPLE_PROMPT_LEN - SAMPLE_GEN_LEN - 1)
        prompt_ids = val_data.data[ix:ix + SAMPLE_PROMPT_LEN].astype(np.int64).tolist()
        sample_prompts_ids.append(prompt_ids)

    # W&B logging
    use_wandb = bool(train_config.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb_config = {
                **model_config.__dict__,
                **{k: v for k, v in train_config.__dict__.items() if not k.startswith("wandb")},
                "n_params": n_params,
                "tokens_per_step": tokens_per_step,
                "total_tokens_trained": total_tokens_trained,
                "train_tokens": train_data.n_tokens,
                "val_tokens": val_data.n_tokens,
                "tokens_per_epoch": tokens_per_epoch,
                "steps_per_epoch": steps_per_epoch,
                "n_epochs": n_epochs,
                "architecture": "llama",
            }
            wandb.init(
                project=train_config.wandb_project,
                name=train_config.wandb_run_name or None,
                config=wandb_config,
                resume="allow" if resume_from else None,
            )
            logger.info("W&B logging enabled: project=%s", train_config.wandb_project)
        except ImportError:
            logger.warning("wandb not installed. Run: pip install wandb")
            use_wandb = False

    # Training loop
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    model.train()

    t0 = time.time()
    current_epoch = tokens_processed / tokens_per_epoch
    prev_epoch_int = int(current_epoch)  # for epoch boundary detection

    for step in range(start_step, total_steps):
        # LR schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y = train_data.get_batch(train_config.micro_batch_size, device)

            with ctx:
                logits, loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        tokens_processed += tokens_per_step
        current_epoch = tokens_processed / tokens_per_epoch

        # Logging
        if step % train_config.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = tokens_processed / dt if dt > 0 else 0
            logger.info(
                "step %5d | loss %.4f | lr %.2e | %.0f tok/s | epoch %.2f | %.1f min elapsed",
                step, loss_accum, lr, tokens_per_sec, current_epoch, dt / 60,
            )
            if use_wandb:
                wandb.log({
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_processed": tokens_processed,
                    "train/epoch": current_epoch,
                    "train/elapsed_min": dt / 60,
                }, step=step)

        # Eval
        if step > 0 and step % train_config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(train_config.eval_steps):
                    x, y = val_data.get_batch(train_config.micro_batch_size, device)
                    with ctx:
                        _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= train_config.eval_steps

            logger.info("step %5d | val_loss %.4f", step, val_loss)

            # Sample generation for quality tracking (still in eval mode)
            samples = []
            with torch.no_grad():
                for prompt_ids in sample_prompts_ids:
                    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                    y = model.generate(x, max_new_tokens=SAMPLE_GEN_LEN, temperature=0.8, top_k=50)
                    gen_ids = y[0].tolist()
                    prompt_text = tokenizer.decode(prompt_ids)
                    full_text = tokenizer.decode(gen_ids)
                    generated_text = full_text[len(prompt_text):]  # strip prompt from output
                    samples.append({
                        "prompt": prompt_text,
                        "generated": generated_text,
                    })
                    logger.info("  [sample] %s → %s", prompt_text[:60], generated_text[:80])

            model.train()

            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/best_loss": min(best_val_loss, val_loss),
                }, step=step)
                # Log samples as a W&B table
                table = wandb.Table(columns=["step", "prompt", "generated"])
                for s in samples:
                    table.add_data(step, s["prompt"], s["generated"])
                wandb.log({"val/samples": table}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = save_dir / "best.pt"
                _save_checkpoint(model, optimizer, model_config, ckpt_path,
                                 step=step, epoch=current_epoch, val_loss=val_loss,
                                 best_val_loss=best_val_loss,
                                 tokens_processed=tokens_processed,
                                 tokens_per_epoch=tokens_per_epoch)
                logger.info("New best! Saved to %s", ckpt_path)

        # Periodic save (every save_interval steps — never overwritten)
        if step > 0 and step % train_config.save_interval == 0:
            ckpt_path = save_dir / f"step_{step:06d}.pt"
            _save_checkpoint(model, optimizer, model_config, ckpt_path,
                             step=step, epoch=current_epoch,
                             val_loss=val_loss if "val_loss" in dir() else None,
                             best_val_loss=best_val_loss,
                             tokens_processed=tokens_processed,
                             tokens_per_epoch=tokens_per_epoch)
            logger.info("Checkpoint saved to %s", ckpt_path)

        # Epoch boundary checkpoint
        curr_epoch_int = int(current_epoch)
        if curr_epoch_int > prev_epoch_int and curr_epoch_int >= 1:
            epoch_ckpt_path = save_dir / f"epoch_{curr_epoch_int:02d}.pt"
            _save_checkpoint(model, optimizer, model_config, epoch_ckpt_path,
                             step=step, epoch=current_epoch,
                             val_loss=val_loss if "val_loss" in dir() else None,
                             best_val_loss=best_val_loss,
                             tokens_processed=tokens_processed,
                             tokens_per_epoch=tokens_per_epoch)
            logger.info("Epoch %d complete! Checkpoint saved to %s", curr_epoch_int, epoch_ckpt_path)
        prev_epoch_int = curr_epoch_int

    # Final save
    total_time = time.time() - t0
    ckpt_path = save_dir / "final.pt"
    _save_checkpoint(model, optimizer, model_config, ckpt_path,
                     step=total_steps, epoch=current_epoch,
                     best_val_loss=best_val_loss,
                     tokens_processed=tokens_processed,
                     tokens_per_epoch=tokens_per_epoch)

    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info("  Total steps:     %d", total_steps)
    logger.info("  Total time:      %.1f hours", total_time / 3600)
    logger.info("  Best val loss:   %.4f", best_val_loss)
    logger.info("  Tokens/sec:      %.0f", tokens_processed / total_time)
    logger.info("  Final checkpoint: %s", ckpt_path)
    logger.info("=" * 70)

    if use_wandb:
        wandb.log({
            "final/best_val_loss": best_val_loss,
            "final/total_hours": total_time / 3600,
            "final/avg_tokens_per_sec": tokens_processed / total_time,
        }, step=total_steps)
        wandb.finish()


# ===========================================================================
# Generation (inference)
# ===========================================================================
def generate_text(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
):
    """Generate text from a trained model."""
    import torch
    import re

    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])

    # Build model
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = MorfessorTokenizer(tokenizer_dir)

    # Segment prompt with Morfessor
    morfessor_model_path = Path(tokenizer_dir).parent / "data" / "morfessor" / "morfessor_telugu.bin"
    if not morfessor_model_path.exists():
        morfessor_model_path = Path("./data/morfessor/morfessor_telugu.bin")

    TELUGU_WORD_RE = re.compile(r"[\u0C00-\u0C7F]+")
    separator = tokenizer.separator

    segmented_prompt = prompt
    if morfessor_model_path.exists():
        try:
            import morfessor
            io = morfessor.MorfessorIO()
            morf_model = io.read_binary_model_file(str(morfessor_model_path))
            tokens = prompt.split()
            seg_tokens = []
            for token in tokens:
                if TELUGU_WORD_RE.fullmatch(token):
                    segments = morf_model.viterbi_segment(token)[0]
                    for i, seg in enumerate(segments):
                        if i < len(segments) - 1:
                            seg_tokens.append(seg + separator)
                        else:
                            seg_tokens.append(seg)
                else:
                    seg_tokens.append(token)
            segmented_prompt = " ".join(seg_tokens)
        except Exception:
            pass

    # Encode
    ids = tokenizer.encode(segmented_prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    # Decode
    generated_ids = y[0].tolist()
    text = tokenizer.decode(generated_ids)

    logger.info("Prompt:    %s", prompt)
    logger.info("Generated: %s", text)
    return text


# ===========================================================================
# Main CLI
# ===========================================================================
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(
        description="Telugu LLaMA — prepare data and train (RoPE + SwiGLU + RMSNorm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data from segmented corpus
  %(prog)s prepare --data ./data/morfessor/segmented_corpus --tokenizer ./tokenizer

  # Train on prepared data
  %(prog)s train --data ./train_data --tokenizer ./tokenizer

  # Both in one go
  %(prog)s all --data ./data/morfessor/segmented_corpus --tokenizer ./tokenizer

  # Generate text from checkpoint
  %(prog)s generate --checkpoint ./checkpoints/best.pt --tokenizer ./tokenizer --prompt "తెలుగు భాష"

  # Resume training
  %(prog)s train --data ./train_data --tokenizer ./tokenizer --resume ./checkpoints/step_005000.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare
    prep = subparsers.add_parser("prepare", help="Tokenize segmented corpus into binary shards")
    prep.add_argument("--data", type=str, required=True, help="Path to segmented corpus directory")
    prep.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
    prep.add_argument("--output", type=str, default="./train_data", help="Output directory for binary shards")
    prep.add_argument("--val-split", type=float, default=0.02, help="Validation split ratio (default: 0.02)")
    prep.add_argument("--workers", type=int, default=0, help="Number of parallel workers (default: auto = cpu_count - 1)")

    # Train
    tr = subparsers.add_parser("train", help="Train LLaMA-style model")
    tr.add_argument("--data", type=str, required=True, help="Path to prepared data (train.bin/val.bin)")
    tr.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
    tr.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    tr.add_argument("--max-steps", type=int, default=45000, help="Max training steps (default: 45000, ~3 epochs)")
    tr.add_argument("--batch-size", type=int, default=32, help="Micro batch size (default: 32)")
    tr.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    tr.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    tr.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    tr.add_argument("--save-interval", type=int, default=500, help="Steps between checkpoints (default: 500)")
    tr.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    tr.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
    tr.add_argument("--wandb", type=str, default="", help="W&B project name (enables logging). E.g. --wandb telugu-gpt")
    tr.add_argument("--wandb-name", type=str, default="", help="W&B run name (optional, auto-generated if empty)")

    # All (prepare + train)
    al = subparsers.add_parser("all", help="Prepare data and train in one go")
    al.add_argument("--data", type=str, required=True, help="Path to segmented corpus directory")
    al.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
    al.add_argument("--output", type=str, default="./train_data", help="Output directory for binary shards")
    al.add_argument("--max-steps", type=int, default=45000, help="Max training steps (~3 epochs)")
    al.add_argument("--batch-size", type=int, default=32, help="Micro batch size")
    al.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    al.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    al.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    al.add_argument("--save-interval", type=int, default=500, help="Steps between checkpoints (default: 500)")
    al.add_argument("--wandb", type=str, default="", help="W&B project name (enables logging)")
    al.add_argument("--wandb-name", type=str, default="", help="W&B run name (optional)")
    al.add_argument("--workers", type=int, default=0, help="Number of parallel workers for data prep (default: auto)")

    # Generate
    gen = subparsers.add_parser("generate", help="Generate text from trained model")
    gen.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    gen.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
    gen.add_argument("--prompt", type=str, required=True, help="Text prompt")
    gen.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    gen.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen.add_argument("--top-k", type=int, default=50, help="Top-k sampling")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    model_config = GPTConfig()
    train_config = TrainConfig()

    if args.command == "prepare":
        prepare_data(
            Path(args.data), Path(args.tokenizer), Path(args.output),
            args.val_split, model_config.block_size,
            num_workers=args.workers,
        )

    elif args.command == "train":
        train_config.micro_batch_size = args.batch_size
        train_config.gradient_accumulation_steps = args.grad_accum
        train_config.learning_rate = args.lr
        train_config.max_steps = args.max_steps
        train_config.lr_decay_steps = args.max_steps
        train_config.save_dir = args.save_dir
        train_config.save_interval = args.save_interval
        train_config.compile_model = not args.no_compile
        train_config.gradient_checkpointing = args.grad_checkpoint
        train_config.wandb_project = args.wandb
        train_config.wandb_run_name = args.wandb_name
        train(Path(args.data), Path(args.tokenizer), model_config, train_config, args.resume)

    elif args.command == "all":
        train_config.micro_batch_size = args.batch_size
        train_config.gradient_accumulation_steps = args.grad_accum
        train_config.learning_rate = args.lr
        train_config.max_steps = args.max_steps
        train_config.lr_decay_steps = args.max_steps
        train_config.save_dir = args.save_dir
        train_config.save_interval = args.save_interval
        train_config.wandb_project = args.wandb
        train_config.wandb_run_name = args.wandb_name
        output_dir = Path(args.output)
        prepare_data(
            Path(args.data), Path(args.tokenizer), output_dir,
            train_config.val_split, model_config.block_size,
            num_workers=args.workers,
        )
        train(output_dir, Path(args.tokenizer), model_config, train_config)

    elif args.command == "generate":
        generate_text(
            Path(args.checkpoint), Path(args.tokenizer),
            args.prompt, args.max_tokens, args.temperature, args.top_k,
        )


if __name__ == "__main__":
    main()
