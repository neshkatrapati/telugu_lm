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
    """LLaMA-style model configuration — ~320M parameters."""
    block_size: int = 2048       # context length
    vocab_size: int = 0          # 0 = auto-detect from tokenizer at runtime
    n_layer: int = 24            # unique transformer layers
    n_head: int = 16             # query attention heads
    n_kv_head: int = 4           # KV head groups for GQA (4 KV heads shared across 16 Q heads)
    n_embd: int = 768            # embedding dimension (v2: 768, v1: 1024)
    dropout: float = 0.0         # 0.0 for pretraining (Liu et al. 2025)
    bias: bool = False           # no bias in linear layers
    rope_theta: float = 10000.0  # RoPE base frequency
    use_weight_sharing: bool = True  # MobileLLM-LS block-wise sharing (2x effective depth)

    # Learned Engrams (conditional memory module)
    use_engrams: bool = False
    engram_table_size: int = 1_000_000     # slots in CPU hash table
    engram_dim: int = 128                   # per-slot embedding dimension
    engram_n_heads: int = 4                 # independent hash heads
    engram_window: int = 6                  # local context window W
    engram_mdl_lambda: float = 0.001        # MDL regularisation weight (gentle)
    engram_mdl_prior: float = 0.3           # Bernoulli prior π for mask sparsity
    engram_inject_indices: tuple = (4, 30)  # schedule indices for injection
    engram_table_lr_mult: float = 5.0       # LR multiplier for memory table
    engram_freeze_table_steps: int = 1000   # freeze table for first N steps
    engram_warmup_steps: int = 3000         # freeze ENTIRE engram module for first N steps

    def effective_depth(self):
        """Number of block forward passes (2x unique layers if weight sharing)."""
        return self.n_layer * 2 if self.use_weight_sharing else self.n_layer

    def param_count(self):
        """Rough parameter count estimate (GQA-aware)."""
        # Embedding: vocab * embd (no positional embedding — using RoPE)
        emb = self.vocab_size * self.n_embd
        head_dim = self.n_embd // self.n_head
        # Attention per layer: Q(n_head*hd*embd) + K(n_kv_head*hd*embd) + V(same) + O(embd^2)
        attn_per_layer = (self.n_head + 2 * self.n_kv_head) * head_dim * self.n_embd + self.n_embd ** 2
        # SwiGLU MLP per layer: gate + up + down = 3 * n_embd * hidden_dim
        hidden_dim = ((int(2 * self.n_embd * 4 / 3) + 255) // 256) * 256
        mlp_per_layer = 3 * self.n_embd * hidden_dim
        # Only unique layers have parameters (weight sharing doesn't add params)
        tfm = self.n_layer * (attn_per_layer + mlp_per_layer)
        # RMSNorm params: (2 per unique layer + 1 final) * n_embd
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
    lr_schedule: str = "wsd"             # "wsd" (warmup-stable-decay) or "cosine"
    wsd_stable_frac: float = 0.7         # WSD: fraction of total steps at peak LR
    wsd_decay_frac: float = 0.2          # WSD: fraction of total steps for decay

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
    tokenizer_type: str = "sp",
    parquet_path: str = None,
):
    """
    Tokenize corpus into memory-mapped binary shards.

    Supports two tokenizer backends:
      - "sp":       SentencePiece (default) — reads raw text or parquet
      - "morfessor": Legacy Morfessor tokenizer — reads .seg.txt files

    Writes train.bin / val.bin as uint32 numpy arrays.
    """
    from tqdm import tqdm

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

    # Stream-write to temp binary (avoids holding all IDs in RAM)
    temp_bin = output_dir / "all_tokens.tmp.bin"
    total_tokens = 0
    total_unk = 0
    BATCH_SIZE = 1000  # lines per micro-batch write

    if tokenizer_type == "sp":
        total_tokens, total_unk = _prepare_sentencepiece(
            data_dir, tokenizer_dir, temp_bin, parquet_path, BATCH_SIZE, tqdm
        )
    else:
        total_tokens, total_unk = _prepare_morfessor(
            data_dir, tokenizer_dir, temp_bin, BATCH_SIZE, tqdm
        )

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

    # Save metadata — get vocab_size from the tokenizer that was used
    vocab_size = _get_vocab_size(tokenizer_dir, tokenizer_type)
    meta = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "train_tokens": int(n_train),
        "val_tokens": int(n_val),
        "total_tokens": int(total_tokens),
        "unk_rate": total_unk / total_tokens if total_tokens else 0,
        "dtype": "uint32",
        "tokenizer_type": tokenizer_type,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return n_train, n_val


def _get_vocab_size(tokenizer_dir: Path, tokenizer_type: str) -> int:
    """Get vocab size from tokenizer without loading the full object."""
    if tokenizer_type == "sp":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_dir / "sp_telugu.model"))
        return sp.get_piece_size()
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from train_tokenizer import MorfessorTokenizer
        return MorfessorTokenizer(tokenizer_dir).vocab_size


def _prepare_sentencepiece(data_dir, tokenizer_dir, temp_bin, parquet_path, batch_size, tqdm):
    """Tokenize raw text using SentencePiece. Returns (total_tokens, total_unk)."""
    import sentencepiece as spm

    sp_model_path = tokenizer_dir / "sp_telugu.model"
    if not sp_model_path.exists():
        logger.error("SentencePiece model not found: %s", sp_model_path)
        logger.error("Train one first: python train_sp_tokenizer.py --input corpus.parquet --output %s", tokenizer_dir)
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    vocab_size = sp.get_piece_size()
    eos_id = sp.eos_id()
    unk_id = sp.unk_id()
    logger.info("Loaded SentencePiece tokenizer: vocab_size=%d", vocab_size)

    total_tokens = 0
    total_unk = 0
    batch_ids = []
    line_count = 0

    def _flush(out_f, batch_ids, total_tokens, total_unk):
        """Write batch to disk and count UNKs."""
        if not batch_ids:
            return total_tokens, total_unk
        arr = np.array(batch_ids, dtype=np.uint32)
        out_f.write(arr.tobytes())
        unk_count = int(np.sum(arr == unk_id))
        return total_tokens + len(arr), total_unk + unk_count

    with open(temp_bin, "wb") as out_f:
        if parquet_path:
            # Stream from parquet
            import pyarrow.parquet as pq
            logger.info("Reading parquet: %s", parquet_path)
            pf = pq.ParquetFile(parquet_path)
            total_rows = pf.metadata.num_rows
            n_row_groups = pf.metadata.num_row_groups

            with tqdm(total=total_rows, desc="Tokenizing (parquet)", unit=" rows") as pbar:
                for rg_idx in range(n_row_groups):
                    table = pf.read_row_group(rg_idx, columns=["text"])
                    col = table.column("text")
                    for text in col.to_pylist():
                        if text is None or not str(text).strip():
                            continue
                        clean = str(text).replace("\n", " ").replace("\r", " ").strip()
                        if not clean:
                            continue
                        ids = sp.encode(clean, out_type=int)
                        ids.append(eos_id)
                        batch_ids.extend(ids)
                        line_count += 1
                        if line_count % batch_size == 0:
                            total_tokens, total_unk = _flush(out_f, batch_ids, total_tokens, total_unk)
                            batch_ids = []
                    pbar.update(len(col))
                    del table, col
        else:
            # Read from text files
            txt_files = sorted(data_dir.rglob("*.txt"))
            if not txt_files:
                logger.error("No .txt files found in %s", data_dir)
                sys.exit(1)
            logger.info("Found %d text files to tokenize", len(txt_files))

            for fpath in txt_files:
                fsize_mb = os.path.getsize(fpath) / 1e6
                logger.info("  Tokenizing %s (%.0f MB)...", fpath.name, fsize_mb)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in tqdm(f, desc=fpath.name, unit=" lines", mininterval=0.5):
                        line = line.strip()
                        if not line:
                            continue
                        ids = sp.encode(line, out_type=int)
                        ids.append(eos_id)
                        batch_ids.extend(ids)
                        line_count += 1
                        if line_count % batch_size == 0:
                            total_tokens, total_unk = _flush(out_f, batch_ids, total_tokens, total_unk)
                            batch_ids = []

        # Final flush
        total_tokens, total_unk = _flush(out_f, batch_ids, total_tokens, total_unk)

    return total_tokens, total_unk


def _prepare_morfessor(data_dir, tokenizer_dir, temp_bin, batch_size, tqdm):
    """Tokenize segmented text using Morfessor tokenizer. Returns (total_tokens, total_unk)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_tokenizer import MorfessorTokenizer

    tokenizer = MorfessorTokenizer(tokenizer_dir)
    logger.info("Loaded Morfessor tokenizer: vocab_size=%d", tokenizer.vocab_size)

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
    _is_tel = tokenizer._is_telugu
    _bpe = tokenizer._encode_token_bpe

    total_tokens = 0
    total_unk = 0

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

                    for token in line.split():
                        # v3: direct lookup — covers ▁, morphemes, BPE, chars
                        tid = _get(token)
                        if tid is not None:
                            batch_ids.append(tid)
                            batch_total += 1
                            if tid == _unk:
                                batch_unk += 1
                            continue

                        # Slow path: token not in vocab
                        batch_total += 1
                        if not _is_tel(token):
                            # Non-Telugu → BPE (cached)
                            sub_ids = _bpe(token)
                            batch_ids.extend(sub_ids)
                            batch_unk += sum(1 for i in sub_ids if i == _unk)
                        else:
                            # Telugu unknown → char fallback
                            for ch in token:
                                cid = _get(ch, _unk)
                                batch_ids.append(cid)
                                if cid == _unk:
                                    batch_unk += 1

                    batch_ids.append(_eos)
                    batch_total += 1
                    line_in_batch += 1

                    if line_in_batch >= batch_size:
                        arr = np.array(batch_ids, dtype=np.uint32)
                        out_f.write(arr.tobytes())
                        total_tokens += batch_total
                        total_unk += batch_unk
                        batch_ids = []
                        batch_unk = 0
                        batch_total = 0
                        line_in_batch = 0

            if batch_ids:
                arr = np.array(batch_ids, dtype=np.uint32)
                out_f.write(arr.tobytes())
                total_tokens += batch_total
                total_unk += batch_unk

    return total_tokens, total_unk


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
        """Apply rotary embeddings to Q and K tensors.

        Handles GQA where xq and xk may have different head counts:
          xq: (B, n_head, T, head_dim)
          xk: (B, n_kv_head, T, head_dim)
          freqs_cis: (T, head_dim//2) complex
        """
        B, Hq, T, D = xq.shape
        _, Hk, _, _ = xk.shape
        xq_ = xq.float().reshape(B, Hq, T, D // 2, 2)
        xk_ = xk.float().reshape(B, Hk, T, D // 2, 2)
        xq_complex = torch.view_as_complex(xq_)
        xk_complex = torch.view_as_complex(xk_)
        freqs = freqs_cis.unsqueeze(0).unsqueeze(0)   # (1, 1, T, D//2)
        xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    # ----- Attention with RoPE + GQA -----
    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            assert config.n_head % config.n_kv_head == 0

            self.n_head = config.n_head
            self.n_kv_head = config.n_kv_head
            self.n_embd = config.n_embd
            self.head_dim = config.n_embd // config.n_head
            self.n_rep = config.n_head // config.n_kv_head  # Q heads per KV group
            self.dropout = config.dropout

            # Separate projections: Q full-rank, K/V reduced for GQA
            self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.resid_dropout = nn.Dropout(config.dropout)

        def forward(self, x, freqs_cis, return_qk=False):
            B, T, C = x.size()

            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

            # Apply RoPE to Q and K (broadcasts correctly for different head counts)
            q, k = apply_rotary_emb(q, k, freqs_cis)

            # Expand KV heads to match Q head count for SDPA
            if self.n_rep > 1:
                k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                k = k.reshape(B, self.n_head, T, self.head_dim)
                v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                v = v.reshape(B, self.n_head, T, self.head_dim)

            if return_qk:
                # Manual attention path (no Flash) — needed to extract QK logits
                scale = 1.0 / math.sqrt(self.head_dim)
                qk = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

                # Causal mask: upper triangle = -inf
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                qk = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                attn_weights = F.softmax(qk, dim=-1)
                if self.training and self.dropout > 0:
                    attn_weights = F.dropout(attn_weights, p=self.dropout)
                y = attn_weights @ v  # (B, n_head, T, head_dim)

                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y, qk  # qk is pre-softmax logits (B, n_head, T, T)
            else:
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

    # ----- Learned Engram Module (conditional memory) -----
    class LearnedEngramModule(nn.Module):
        """Attention-derived conditional memory lookup with MDL regularisation.

        Takes hidden states + QK attention logits, selects a local window
        of token IDs via Gumbel-sigmoid masks, hashes into a CPU-resident
        memory table, and returns a gated additive residual.

        Zero-initialised so the module starts as identity (no contribution).
        """

        def __init__(self, config):
            super().__init__()
            W = config.engram_window          # 6
            n_embd = config.n_embd            # 768
            engram_dim = config.engram_dim    # 128
            n_heads = config.engram_n_heads   # 4

            self.W = W
            self.n_embd = n_embd
            self.engram_dim = engram_dim
            self.n_heads = n_heads
            self.head_dim = engram_dim // n_heads  # 32

            # MDL hyperparams
            self.mdl_lambda = config.engram_mdl_lambda
            self.mdl_prior = config.engram_mdl_prior

            # Pattern predictor: QK window logits → binary mask logits
            self.pattern_mlp = nn.Sequential(
                nn.Linear(W, 2 * W),
                nn.SiLU(),
                nn.Linear(2 * W, W),
            )

            # Memory → hidden projections
            self.W_K = nn.Linear(engram_dim, n_embd, bias=False)
            self.W_V = nn.Linear(engram_dim, n_embd, bias=False)

            # Layer norms for gating
            self.ln_k = RMSNorm(n_embd)
            self.ln_h = RMSNorm(n_embd)

            # Depthwise causal convolution
            self.conv = nn.Conv1d(
                n_embd, n_embd, kernel_size=4,
                padding=3, groups=n_embd,  # depthwise
            )
            self.conv_act = nn.SiLU()

            # Hashing: 4 primes near 250K for multi-head hashing
            hash_primes = torch.tensor([249989, 249973, 249961, 249947], dtype=torch.long)
            self.register_buffer("hash_primes", hash_primes)
            # Random multiplicative weights per (head, position)
            hash_weights = torch.randint(1, 1_000_000, (n_heads, W), dtype=torch.long)
            self.register_buffer("hash_weights", hash_weights)

            # Sentinel offset for padding positions
            self.sentinel_base = config.vocab_size + 1

            # Zero-init: module starts as identity (no contribution)
            nn.init.zeros_(self.W_V.weight)
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

        @torch.compiler.disable  # CPU↔GPU memory table lookup breaks Dynamo tracing
        def forward(self, h, qk_logits, token_ids, memory_table, tau=1.0):
            """
            Args:
                h:            (B, T, n_embd) — hidden states after attention
                qk_logits:    (B, n_head, T, T) — pre-softmax QK logits
                token_ids:    (B, T) — input token IDs (int)
                memory_table: nn.Embedding on CPU (table_size, engram_dim)
                tau:          Gumbel temperature for mask sampling

            Returns:
                engram_out:   (B, T, n_embd) — additive residual
                mdl_loss:     scalar — MDL regularisation loss
            """
            B, T, C = h.shape
            W = self.W
            device = h.device

            # --- 1. Build window token IDs: (B, T, W) ---
            # For position p, gather token_ids[:, p-W:p]
            # Sentinel values for positions that fall before the sequence start
            sentinels = torch.arange(W, 0, -1, device=device, dtype=token_ids.dtype)
            sentinels = self.sentinel_base + sentinels  # (W,) unique sentinel IDs

            # Pad token_ids with sentinels at the front
            # sentinel_pad shape: (B, W)
            sentinel_pad = sentinels.unsqueeze(0).expand(B, -1)
            padded_ids = torch.cat([sentinel_pad, token_ids], dim=1)  # (B, T+W)

            # Gather windows: for each position p in [0, T), gather [p, p+1, ..., p+W-1]
            # which corresponds to original positions [p-W, p-W+1, ..., p-1]
            offsets = torch.arange(W, device=device).unsqueeze(0)  # (1, W)
            positions = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
            gather_idx = positions + offsets  # (T, W)
            gather_idx = gather_idx.unsqueeze(0).expand(B, -1, -1)  # (B, T, W)
            window_ids = torch.gather(
                padded_ids.unsqueeze(1).expand(-1, T, -1),
                dim=2, index=gather_idx
            )  # (B, T, W) -- but this is wrong, let me fix

            # Actually simpler: just index padded_ids directly
            # padded_ids is (B, T+W). For position p (0-indexed in original),
            # the window is padded_ids[:, p:p+W]
            window_ids = padded_ids.unfold(1, W, 1)[:, :T, :]  # (B, T, W)

            # --- 2. Extract QK window logits: (B, T, W) ---
            # qk_logits is (B, n_head, T, T). For each query position p,
            # gather the W key positions [p-W, ..., p-1] → average over heads
            # Build gather indices for keys (same padding logic)
            # Key positions for query p: max(0, p-W) to p-1
            # In padded indexing: positions [p, p+1, ..., p+W-1] in the original T dim
            # But QK is (T, T) not padded. We need to handle boundary carefully.

            # For each query p, the W key indices are [p-W, ..., p-1]
            # Clamp negatives to 0 and use mask for OOB
            key_indices = positions - W + offsets  # (T, W), values in [-W+1, T-1]
            valid_mask = key_indices >= 0  # (T, W)
            key_indices_clamped = key_indices.clamp(min=0)  # (T, W)

            # Gather from qk_logits: (B, n_head, T, T) → need (B, n_head, T, W)
            ki_expanded = key_indices_clamped.unsqueeze(0).unsqueeze(0).expand(
                B, qk_logits.size(1), -1, -1
            )  # (B, n_head, T, W)
            qk_window = torch.gather(qk_logits, dim=3, index=ki_expanded)  # (B, n_head, T, W)

            # Mask out-of-bounds positions (before sequence start)
            valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand_as(qk_window)
            qk_window = qk_window.masked_fill(~valid_mask_expanded, 0.0)

            # Average across attention heads → (B, T, W)
            qk_window = qk_window.mean(dim=1)

            # --- 3. Pattern predictor → mask logits ---
            mask_logits = self.pattern_mlp(qk_window)  # (B, T, W)

            # --- 4. Gumbel-sigmoid for differentiable binary masks ---
            p_w = torch.sigmoid(mask_logits)  # soft mask probabilities

            if self.training:
                # Gumbel-sigmoid with straight-through estimator
                u = torch.rand_like(mask_logits).clamp(1e-6, 1 - 1e-6)
                gumbel_noise = torch.log(u) - torch.log(1 - u)
                y_soft = torch.sigmoid((mask_logits + gumbel_noise) / tau)
                hard_mask = (y_soft > 0.5).float()
                # Straight-through: forward uses hard, backward uses soft
                hard_mask = hard_mask - y_soft.detach() + y_soft
            else:
                hard_mask = (p_w > 0.5).float()

            # --- 5. MDL loss: KL(Bernoulli(p_w) || Bernoulli(π)) ---
            pi = self.mdl_prior
            # KL = p * log(p/π) + (1-p) * log((1-p)/(1-π))
            eps = 1e-7
            p_clamped = p_w.clamp(eps, 1 - eps)
            kl = (p_clamped * (p_clamped / pi).log()
                  + (1 - p_clamped) * ((1 - p_clamped) / (1 - pi)).log())
            mdl_loss = self.mdl_lambda * kl.mean()

            # --- 6. Form hash keys ---
            # Use sentinel values for masked-out positions
            sentinel_vals = sentinels.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # (B, T, W)
            keys = (window_ids * hard_mask.long()
                    + sentinel_vals * (1 - hard_mask).long())  # (B, T, W)

            # --- 7. Multi-head hashing → memory lookup ---
            # For each head k: idx_k = (Σ_w keys[:,w] * hash_weights[k,w]) % hash_primes[k]
            # keys: (B, T, W), hash_weights: (n_heads, W) → (B, T, n_heads)
            keys_long = keys.long()
            # (B, T, W) × (n_heads, W).T → sum over W → (B, T, n_heads)
            hash_vals = torch.einsum("btw,hw->bth", keys_long.float(),
                                      self.hash_weights.float()).long()
            # Modulo by per-head primes
            indices = hash_vals % self.hash_primes.unsqueeze(0).unsqueeze(0)  # (B, T, n_heads)
            # Clamp to table size
            indices = indices.clamp(0, memory_table.weight.size(0) - 1)

            # Flatten, gather unique indices from CPU table, move to GPU
            flat_indices = indices.reshape(-1)  # (B*T*n_heads,)

            # Gather from CPU memory table
            unique_indices, inverse = flat_indices.unique(return_inverse=True)
            unique_embeddings = memory_table(unique_indices.cpu()).to(device)  # (n_unique, engram_dim)
            gathered = unique_embeddings[inverse]  # (B*T*n_heads, engram_dim)

            # Each head k looked up a full engram_dim vector; extract head k's
            # own head_dim slice and concatenate across heads → (B, T, engram_dim)
            gathered = gathered.view(B, T, self.n_heads, self.engram_dim)  # (B, T, 4, 128)
            # View engram_dim as (n_heads, head_dim) → (B, T, 4, 4, 32)
            gathered = gathered.view(B, T, self.n_heads, self.n_heads, self.head_dim)
            # Diagonal: head k uses the k-th head_dim slice from its lookup
            head_range = torch.arange(self.n_heads, device=device)
            e = gathered[:, :, head_range, head_range, :]  # (B, T, 4, 32)
            e = e.reshape(B, T, self.engram_dim)  # (B, T, 128)

            # --- 8. Project + gate ---
            k_t = self.W_K(e)   # (B, T, n_embd)
            v_t = self.W_V(e)   # (B, T, n_embd)

            # Gate: α = sigmoid(dot(ln_h(h), ln_k(k_t)) / sqrt(n_embd))
            h_normed = self.ln_h(h)
            k_normed = self.ln_k(k_t)
            alpha = torch.sigmoid(
                (h_normed * k_normed).sum(dim=-1, keepdim=True)
                / math.sqrt(self.n_embd)
            )  # (B, T, 1)
            v_hat = alpha * v_t  # (B, T, n_embd)

            # --- 9. Causal convolution ---
            # (B, T, C) → (B, C, T) → conv → trim to T → (B, C, T) → (B, T, C)
            conv_in = v_hat.transpose(1, 2)  # (B, C, T)
            conv_out = self.conv(conv_in)[:, :, :T]  # trim for causal (padding=3, kernel=4)
            conv_out = self.conv_act(conv_out)
            conv_out = conv_out.transpose(1, 2)  # (B, T, C)

            # Residual: conv output + gated value
            out = conv_out + v_hat

            return out, mdl_loss

    # ----- Transformer Block -----
    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = RMSNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = RMSNorm(config.n_embd)
            self.mlp = SwiGLUMLP(config)

        def forward(self, x, freqs_cis, engram_ctx=None):
            if engram_ctx is not None:
                engram_mod, token_ids, memory_table, tau = engram_ctx
                attn_out, qk_logits = self.attn(self.ln_1(x), freqs_cis, return_qk=True)
                engram_out, mdl_loss = engram_mod(
                    x + attn_out, qk_logits, token_ids, memory_table, tau
                )
                x = x + attn_out + engram_out
                x = x + self.mlp(self.ln_2(x))
                return x, mdl_loss
            else:
                x = x + self.attn(self.ln_1(x), freqs_cis)
                x = x + self.mlp(self.ln_2(x))
                return x

    # ----- GPT (LLaMA-style with GQA + weight sharing) -----
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

            # Block-wise weight sharing: each unique block runs twice
            # _block_schedule is a plain list of module references (not nn.ModuleList)
            if config.use_weight_sharing:
                self._block_schedule = []
                for block in self.transformer.h:
                    self._block_schedule.append(block)
                    self._block_schedule.append(block)
            else:
                self._block_schedule = list(self.transformer.h)

            self._effective_depth = len(self._block_schedule)

            # Learned Engrams: CPU memory table + per-injection-point modules
            self._engram_map = {}  # schedule_index → engram_module_index
            if config.use_engrams:
                # CPU-resident memory table (~512 MB for 1M × 128 fp32)
                self.memory_table = nn.Embedding(
                    config.engram_table_size, config.engram_dim, sparse=True
                )
                nn.init.zeros_(self.memory_table.weight)
                # Keep on CPU — gathered per-step, moved to GPU
                self.memory_table = self.memory_table.cpu()

                inject_indices = config.engram_inject_indices or (4, 30)
                self.engram_modules = nn.ModuleList([
                    LearnedEngramModule(config) for _ in inject_indices
                ])
                self._engram_map = {idx: i for i, idx in enumerate(inject_indices)}

            # Precompute RoPE frequencies and store as buffer
            head_dim = config.n_embd // config.n_head
            freqs_cis = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
            self.register_buffer("freqs_cis", torch.view_as_real(freqs_cis))

            # Init weights
            self.apply(self._init_weights)
            # Scale residual projections by effective depth (attention c_proj + SwiGLU w_down)
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self._effective_depth))

            # Re-zero engram W_V and conv after _init_weights (which would have overwritten)
            if config.use_engrams:
                for em in self.engram_modules:
                    nn.init.zeros_(em.W_V.weight)
                    nn.init.zeros_(em.conv.weight)
                    nn.init.zeros_(em.conv.bias)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                pass  # weight already initialized to ones in constructor

        def gradient_checkpointing_enable(self):
            """Enable gradient checkpointing to trade compute for memory."""
            self._gradient_checkpointing = True

        def gradient_checkpointing_disable(self):
            self._gradient_checkpointing = False

        def forward(self, idx, targets=None):
            device = idx.device
            B, T = idx.size()
            assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

            tok_emb = self.transformer.wte(idx)
            x = self.transformer.drop(tok_emb)

            # Recover complex freqs from stored real buffer, slice to seq length
            freqs_cis = torch.view_as_complex(self.freqs_cis[:T])

            use_ckpt = getattr(self, "_gradient_checkpointing", False) and self.training
            # Use _engram_map_active (empty during warmup, populated after)
            active_map = getattr(self, "_engram_map_active", self._engram_map)
            use_engrams = self.config.use_engrams and active_map
            mdl_losses = []

            for sched_idx, block in enumerate(self._block_schedule):
                engram_ctx = None
                if use_engrams and sched_idx in active_map:
                    mod_idx = active_map[sched_idx]
                    tau = getattr(self, "_current_tau", 0.3)
                    engram_ctx = (
                        self.engram_modules[mod_idx],
                        idx, self.memory_table, tau,
                    )

                if engram_ctx is not None:
                    # Engram blocks: manual attention + memory lookup
                    # (not compatible with gradient checkpointing due to return tuple)
                    x, mdl_loss = block(x, freqs_cis, engram_ctx=engram_ctx)
                    mdl_losses.append(mdl_loss)
                else:
                    if use_ckpt:
                        x = torch.utils.checkpoint.checkpoint(
                            block, x, freqs_cis, use_reentrant=False,
                        )
                    else:
                        x = block(x, freqs_cis)

            x = self.transformer.ln_f(x)

            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                if mdl_losses:
                    self._last_mdl_loss = sum(mdl_losses).item()
                    loss = loss + sum(mdl_losses)
                else:
                    self._last_mdl_loss = 0.0
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
def _save_checkpoint(model, optimizer, model_config, path, table_optimizer=None, **kwargs):
    """Save a checkpoint with full metadata. Never loses data."""
    import torch
    state = model.state_dict() if not hasattr(model, "_orig_mod") else model._orig_mod.state_dict()
    ckpt = {
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": model_config.__dict__,
        "architecture": "llama",
    }
    if table_optimizer is not None:
        ckpt["table_optimizer"] = table_optimizer.state_dict()
    ckpt.update(kwargs)
    torch.save(ckpt, str(path))
    del ckpt, state
    torch.cuda.empty_cache()


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

    # If resuming, peek at checkpoint config to override model architecture
    # so build_model creates a compatible model. Keep checkpoint in memory
    # to avoid loading it twice.
    _pending_checkpoint = None
    if resume_from and os.path.exists(resume_from):
        logger.info("Loading checkpoint %s (to CPU)...", resume_from)
        _pending_checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        ckpt_cfg = _pending_checkpoint.get("config", {})
        if ckpt_cfg:
            overrides = []
            for key in ("n_layer", "n_head", "n_kv_head", "n_embd", "use_weight_sharing",
                        "dropout", "block_size", "rope_theta", "bias"):
                if key in ckpt_cfg and getattr(model_config, key) != ckpt_cfg[key]:
                    overrides.append(f"{key}: {getattr(model_config, key)} → {ckpt_cfg[key]}")
                    setattr(model_config, key, ckpt_cfg[key])
            if overrides:
                logger.info("Overriding model config from checkpoint: %s", ", ".join(overrides))

    # Build model
    model = build_model(model_config, device)
    n_params = model.count_parameters()
    logger.info("  Model params: %d (%.1fM)", n_params, n_params / 1e6)
    logger.info("  Block size:   %d", model_config.block_size)
    logger.info("  Layers:       %d unique, %d effective", model_config.n_layer, model_config.effective_depth())
    logger.info("  Q Heads:      %d", model_config.n_head)
    logger.info("  KV Heads:     %d (GQA ratio: %d Q per KV)", model_config.n_kv_head,
                model_config.n_head // model_config.n_kv_head)
    logger.info("  Embed dim:    %d", model_config.n_embd)
    logger.info("  Weight share: %s", model_config.use_weight_sharing)
    logger.info("  Dropout:      %.2f", model_config.dropout)
    logger.info("  LR schedule:  %s", train_config.lr_schedule)

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
    # Keep engram memory table on CPU (too large for GPU, ~512MB for 1M×128)
    if model_config.use_engrams and hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()
        logger.info("Engram memory table pinned to CPU (%d × %d = %.0f MB)",
                     model_config.engram_table_size, model_config.engram_dim,
                     model_config.engram_table_size * model_config.engram_dim * 4 / 1e6)

    # Resume BEFORE compile — checkpoint has raw keys (no _orig_mod. prefix)
    start_step = 0
    best_val_loss = float("inf")
    tokens_processed = 0
    _resume_optimizer_state = None
    _resume_table_optimizer_state = None
    if _pending_checkpoint is not None:
        checkpoint = _pending_checkpoint
        _pending_checkpoint = None
        model.load_state_dict(checkpoint["model"])
        # Ensure memory table stays on CPU after state_dict load
        if model_config.use_engrams and hasattr(model, "memory_table"):
            model.memory_table = model.memory_table.cpu()
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        tokens_processed = checkpoint.get("tokens_processed", start_step * tokens_per_step)
        _resume_optimizer_state = checkpoint.get("optimizer")
        _resume_table_optimizer_state = checkpoint.get("table_optimizer")
        logger.info("Resumed at step %d (epoch %.2f), best_val_loss=%.4f",
                     start_step, tokens_processed / tokens_per_epoch, best_val_loss)
        del checkpoint  # free CPU memory immediately

    # Gradient checkpointing (enable BEFORE compile so the checkpoint calls
    # are part of the graph that torch.compile sees)
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled — activation memory will be O(sqrt(layers))")

    # Compile (after loading weights so keys match)
    if train_config.compile_model and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer — separate SparseAdam for memory table (sparse gradients)
    _raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    param_dict = {pn: p for pn, p in _raw_model.named_parameters() if p.requires_grad}
    # Separate memory table params from the rest
    table_params = []
    decay_params = []
    nodecay_params = []
    for pn, p in param_dict.items():
        if "memory_table" in pn:
            table_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    use_fused = (device == "cuda")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=use_fused,
    )

    # Sparse optimizer for CPU memory table (nn.Embedding(sparse=True) → sparse grads)
    table_optimizer = None
    if model_config.use_engrams and table_params:
        table_optimizer = torch.optim.SparseAdam(
            table_params,
            lr=train_config.learning_rate * model_config.engram_table_lr_mult,
            betas=(train_config.beta1, train_config.beta2),
        )
        logger.info("Engram memory table: %d params, lr_mult=%.1f, SparseAdam",
                     sum(p.numel() for p in table_params), model_config.engram_table_lr_mult)

    # Load optimizer state if resuming (from state saved earlier, no second disk load)
    if _resume_optimizer_state is not None:
        optimizer.load_state_dict(_resume_optimizer_state)
        logger.info("Restored optimizer state")
        del _resume_optimizer_state
        torch.cuda.empty_cache()
    if _resume_table_optimizer_state is not None and table_optimizer is not None:
        table_optimizer.load_state_dict(_resume_table_optimizer_state)
        logger.info("Restored table optimizer state")
        del _resume_table_optimizer_state

    # LR schedule
    def get_lr(step):
        if train_config.lr_schedule == "wsd":
            # WSD: Warmup -> Stable -> Decay (MiniCPM-style, resumable)
            total = train_config.max_steps
            warmup_end = train_config.warmup_steps
            decay_steps = int(total * train_config.wsd_decay_frac)
            decay_start = total - decay_steps

            if step < warmup_end:
                return train_config.learning_rate * step / warmup_end
            elif step < decay_start:
                return train_config.learning_rate
            else:
                if decay_steps <= 0:
                    return train_config.min_lr
                decay_ratio = (step - decay_start) / decay_steps
                return train_config.min_lr + (1.0 - decay_ratio) * (train_config.learning_rate - train_config.min_lr)
        else:
            # Original cosine schedule
            if step < train_config.warmup_steps:
                return train_config.learning_rate * step / train_config.warmup_steps
            if step > train_config.lr_decay_steps:
                return train_config.min_lr
            decay_ratio = (step - train_config.warmup_steps) / (train_config.lr_decay_steps - train_config.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

    # Save dir
    save_dir = Path(train_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for sample generation during eval
    sp_model_path = tokenizer_dir / "sp_telugu.model"
    if sp_model_path.exists():
        import sentencepiece as spm
        _sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        decode_fn = lambda ids: _sp.decode(ids)
        logger.info("Loaded SentencePiece tokenizer for decoding")
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from train_tokenizer import MorfessorTokenizer
        _morf_tok = MorfessorTokenizer(tokenizer_dir)
        decode_fn = _morf_tok.decode
        logger.info("Loaded Morfessor tokenizer for decoding")

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
        # Update table optimizer LR (separate SparseAdam)
        if table_optimizer is not None:
            for param_group in table_optimizer.param_groups:
                param_group["lr"] = lr * model_config.engram_table_lr_mult

        # Engram: warmup (full freeze) + temperature annealing + table freezing
        if model_config.use_engrams:
            engram_active = step >= model_config.engram_warmup_steps

            # During warmup: freeze entire engram module + table (pure base model training)
            for em in _raw_model.engram_modules.parameters():
                em.requires_grad_(engram_active)
            _raw_model.memory_table.weight.requires_grad_(
                engram_active and step >= model_config.engram_freeze_table_steps
            )

            # Disable engram injection during warmup by clearing the map
            if not engram_active:
                _raw_model._engram_map_active = {}
            else:
                _raw_model._engram_map_active = _raw_model._engram_map

            # Temperature annealing (only matters once active)
            if engram_active:
                steps_since_active = step - model_config.engram_warmup_steps
                remaining_steps = total_steps - model_config.engram_warmup_steps
                anneal_end = int(remaining_steps * 0.3)
                tau = max(0.3, 1.0 - 0.7 * steps_since_active / anneal_end) if steps_since_active < anneal_end else 0.3
            else:
                tau = 1.0
            _raw_model._current_tau = tau

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        if table_optimizer is not None:
            table_optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y = train_data.get_batch(train_config.micro_batch_size, device)

            with ctx:
                logits, loss = model(x, y)
                loss = loss / train_config.gradient_accumulation_steps

            if step <= start_step + 1 and micro_step == 0:
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                logger.info("  [MEM] step=%d micro=%d BEFORE backward: alloc=%.1fG reserved=%.1fG",
                            step, micro_step, alloc, resv)
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradient clipping (exclude sparse memory table — norm op unsupported on SparseCPU)
        scaler.unscale_(optimizer)
        clip_params = [p for p in model.parameters() if p.grad is not None and not p.grad.is_sparse]
        torch.nn.utils.clip_grad_norm_(clip_params, train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Step sparse optimizer for memory table (not affected by GradScaler)
        if table_optimizer is not None:
            table_optimizer.step()

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
                log_dict = {
                    "train/loss": loss_accum,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_processed": tokens_processed,
                    "train/epoch": current_epoch,
                    "train/elapsed_min": dt / 60,
                }
                if model_config.use_engrams:
                    log_dict["engram/mdl_loss"] = getattr(_raw_model, "_last_mdl_loss", 0.0)
                    log_dict["engram/tau"] = getattr(_raw_model, "_current_tau", 0.3)
                    log_dict["engram/active"] = 1.0 if step >= model_config.engram_warmup_steps else 0.0
                wandb.log(log_dict, step=step)

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
            # Only run if torch.compile is NOT active — dynamo caches from
            # variable-length generate() cause OOM on the next backward().
            samples = []
            is_compiled = hasattr(model, "_orig_mod")
            if not is_compiled:
                gen_model = model
                with torch.no_grad():
                    for prompt_ids in sample_prompts_ids:
                        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                        y = gen_model.generate(x, max_new_tokens=SAMPLE_GEN_LEN, temperature=0.8, top_k=50)
                        gen_ids = y[0].tolist()
                        del x, y
                        prompt_text = decode_fn(prompt_ids)
                        full_text = decode_fn(gen_ids)
                        generated_text = full_text[len(prompt_text):]
                        samples.append({
                            "prompt": prompt_text,
                            "generated": generated_text,
                        })
                        logger.info("  [sample] %s → %s", prompt_text[:60], generated_text[:80])
                torch.cuda.empty_cache()

            model.train()
            alloc = torch.cuda.memory_allocated() / 1e9
            resv = torch.cuda.memory_reserved() / 1e9
            logger.info("  [MEM] after eval+save, back to train: alloc=%.1fG reserved=%.1fG", alloc, resv)

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
                                 table_optimizer=table_optimizer,
                                 step=step, epoch=current_epoch, val_loss=val_loss,
                                 best_val_loss=best_val_loss,
                                 tokens_processed=tokens_processed,
                                 tokens_per_epoch=tokens_per_epoch)
                logger.info("New best! Saved to %s", ckpt_path)

        # Periodic save (every save_interval steps — never overwritten)
        if step > 0 and step % train_config.save_interval == 0:
            ckpt_path = save_dir / f"step_{step:06d}.pt"
            _save_checkpoint(model, optimizer, model_config, ckpt_path,
                             table_optimizer=table_optimizer,
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
                             table_optimizer=table_optimizer,
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
                     table_optimizer=table_optimizer,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    config = GPTConfig(
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg.get("n_kv_head", cfg["n_head"]),
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.0),
        bias=cfg["bias"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        use_weight_sharing=cfg.get("use_weight_sharing", False),
        # Learned Engrams (restored from checkpoint if present)
        use_engrams=cfg.get("use_engrams", False),
        engram_table_size=cfg.get("engram_table_size", 1_000_000),
        engram_dim=cfg.get("engram_dim", 128),
        engram_n_heads=cfg.get("engram_n_heads", 4),
        engram_window=cfg.get("engram_window", 6),
        engram_inject_indices=tuple(cfg.get("engram_inject_indices", (4, 30))),
    )

    # Build model
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    if config.use_engrams and hasattr(model, "memory_table"):
        model.memory_table = model.memory_table.cpu()

    # Load tokenizer (SentencePiece or Morfessor)
    sp_model_path = tokenizer_dir / "sp_telugu.model"
    if sp_model_path.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
        decode_fn = lambda tok_ids: sp.decode(tok_ids)
    else:
        import re
        sys.path.insert(0, str(Path(__file__).parent))
        from train_tokenizer import MorfessorTokenizer
        tokenizer = MorfessorTokenizer(tokenizer_dir)
        decode_fn = tokenizer.decode

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

        ids = tokenizer.encode(segmented_prompt, add_bos=True, add_eos=False)

    x = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    # Decode
    generated_ids = y[0].tolist()
    text = decode_fn(generated_ids)

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
    prep = subparsers.add_parser("prepare", help="Tokenize corpus into binary shards")
    prep.add_argument("--data", type=str, default=None, help="Path to corpus directory (text files or segmented)")
    prep.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
    prep.add_argument("--output", type=str, default="./train_data", help="Output directory for binary shards")
    prep.add_argument("--val-split", type=float, default=0.02, help="Validation split ratio (default: 0.02)")
    prep.add_argument("--tokenizer-type", type=str, default="sp", choices=["sp", "morfessor"],
                       help="Tokenizer backend (default: sp = SentencePiece)")
    prep.add_argument("--parquet", type=str, default=None, help="Parquet file path (alternative to --data for raw text)")
    prep.add_argument("--workers", type=int, default=0, help="Number of parallel workers (default: auto)")

    # Shared architecture + training args (added to both train and all)
    def _add_train_args(p):
        """Add training arguments shared between 'train' and 'all' subcommands."""
        p.add_argument("--data", type=str, required=True, help="Path to prepared data (train.bin/val.bin)")
        p.add_argument("--tokenizer", type=str, default="./tokenizer", help="Tokenizer directory")
        p.add_argument("--max-steps", type=int, default=45000, help="Max training steps (default: 45000)")
        p.add_argument("--batch-size", type=int, default=32, help="Micro batch size (default: 32)")
        p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
        p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
        p.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint directory")
        p.add_argument("--save-interval", type=int, default=500, help="Steps between checkpoints (default: 500)")
        p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
        p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
        p.add_argument("--wandb", type=str, default="", help="W&B project name (enables logging)")
        p.add_argument("--wandb-name", type=str, default="", help="W&B run name (optional)")
        # Architecture
        p.add_argument("--n-layer", type=int, default=24, help="Number of unique transformer layers (default: 24)")
        p.add_argument("--n-embd", type=int, default=768, help="Embedding dimension (default: 768)")
        p.add_argument("--n-kv-head", type=int, default=4, help="KV head groups for GQA (default: 4)")
        p.add_argument("--no-weight-sharing", action="store_true", help="Disable block-wise weight sharing")
        p.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)")
        # Learned Engrams
        p.add_argument("--use-engrams", action="store_true", help="Enable learned engram memory module")
        p.add_argument("--engram-table-size", type=int, default=1_000_000,
                       help="Engram hash table slots (default: 1M)")
        p.add_argument("--engram-freeze-steps", type=int, default=1000,
                       help="Freeze engram table for first N steps after warmup (default: 1000)")
        p.add_argument("--engram-warmup-steps", type=int, default=3000,
                       help="Freeze entire engram module for first N steps (default: 3000)")
        # LR schedule
        p.add_argument("--lr-schedule", type=str, default="wsd", choices=["wsd", "cosine"],
                       help="LR schedule (default: wsd)")
        p.add_argument("--wsd-stable-frac", type=float, default=0.7,
                       help="WSD: fraction of steps at stable LR (default: 0.7)")
        p.add_argument("--wsd-decay-frac", type=float, default=0.2,
                       help="WSD: fraction of steps for decay (default: 0.2)")

    # Train
    tr = subparsers.add_parser("train", help="Train LLaMA-style model")
    _add_train_args(tr)
    tr.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    # All (prepare + train) — --data is optional if --parquet is given
    al = subparsers.add_parser("all", help="Prepare data and train in one go")
    _add_train_args(al)
    # Override --data to not be required (parquet can substitute for input)
    for action in al._actions:
        if hasattr(action, 'dest') and action.dest == 'data':
            action.required = False
            break
    al.add_argument("--output", type=str, default="./train_data", help="Output for binary shards")
    al.add_argument("--tokenizer-type", type=str, default="sp", choices=["sp", "morfessor"],
                     help="Tokenizer backend (default: sp)")
    al.add_argument("--parquet", type=str, default=None, help="Parquet file path")
    al.add_argument("--workers", type=int, default=0, help="Parallel workers for data prep")

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

    def _apply_train_args(args, model_config, train_config):
        """Apply shared training CLI args to config objects."""
        model_config.n_layer = args.n_layer
        model_config.n_embd = args.n_embd
        model_config.n_kv_head = args.n_kv_head
        model_config.use_weight_sharing = not args.no_weight_sharing
        model_config.dropout = args.dropout
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
        train_config.lr_schedule = args.lr_schedule
        train_config.wsd_stable_frac = args.wsd_stable_frac
        train_config.wsd_decay_frac = args.wsd_decay_frac
        # Learned Engrams
        model_config.use_engrams = args.use_engrams
        model_config.engram_table_size = args.engram_table_size
        model_config.engram_freeze_table_steps = args.engram_freeze_steps
        model_config.engram_warmup_steps = args.engram_warmup_steps

    if args.command == "prepare":
        if not args.data and not args.parquet:
            parser.error("prepare requires either --data (text dir) or --parquet (parquet file)")
        data_dir = Path(args.data) if args.data else Path(".")
        prepare_data(
            data_dir, Path(args.tokenizer), Path(args.output),
            args.val_split, model_config.block_size,
            num_workers=args.workers,
            tokenizer_type=args.tokenizer_type,
            parquet_path=args.parquet,
        )

    elif args.command == "train":
        _apply_train_args(args, model_config, train_config)
        train(Path(args.data), Path(args.tokenizer), model_config, train_config, args.resume)

    elif args.command == "all":
        _apply_train_args(args, model_config, train_config)
        if not args.data and not args.parquet:
            parser.error("'all' requires either --data (text dir) or --parquet (parquet file)")
        data_dir = Path(args.data) if args.data else Path(".")
        output_dir = Path(args.output)
        prepare_data(
            data_dir, Path(args.tokenizer), output_dir,
            train_config.val_split, model_config.block_size,
            num_workers=args.workers,
            tokenizer_type=args.tokenizer_type,
            parquet_path=args.parquet,
        )
        train(output_dir, Path(args.tokenizer), model_config, train_config)

    elif args.command == "generate":
        generate_text(
            Path(args.checkpoint), Path(args.tokenizer),
            args.prompt, args.max_tokens, args.temperature, args.top_k,
        )


if __name__ == "__main__":
    main()
