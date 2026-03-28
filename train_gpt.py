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

    # Engrams v3 (PMI-based 5-gram pattern memory)
    use_engrams: bool = False
    engram_table_size: int = 580_262        # number of unique patterns from preprocessing
    engram_dim: int = 64                    # per-pattern embedding dimension
    engram_inject_indices: tuple = (4, 30)  # schedule indices for injection
    engram_table_lr_mult: float = 5.0       # LR multiplier for pattern table
    engram_warmup_steps: int = 3000         # freeze ENTIRE engram module for first N steps
    engram_gate_freeze_steps: int = 500      # force gate open (0.5) for N steps after warmup

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

        def forward(self, x, freqs_cis):
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

    # ----- Engram Module v3 (PMI-based 5-gram pattern memory) -----
    class PatternEngramModule(nn.Module):
        """Pattern-based engram memory using precomputed 5-gram templates.

        Each token position has a precomputed pattern_id from PMI-scored
        5-gram templates (exact collocations + partial templates). The
        pattern_id indexes into a learned embedding table on GPU.

        Gate is initialized slightly closed (sigmoid(-2) ≈ 0.12) so the
        module starts near-identity. Positions with pattern_id == -1
        (no match) get a zero vector.
        """

        def __init__(self, config):
            super().__init__()
            n_embd = config.n_embd
            engram_dim = config.engram_dim

            self.n_embd = n_embd
            self.engram_dim = engram_dim

            # Pattern embedding → hidden projection
            self.W_V = nn.Linear(engram_dim, n_embd, bias=False)

            # Gate: scalar per position via projection
            self.ln_gate = RMSNorm(n_embd)
            self.gate_proj = nn.Linear(n_embd, 1, bias=True)

            # Gate starts slightly closed (bias=-2 → sigmoid≈0.12)
            nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.gate_proj.bias, -0.5)

        def forward(self, h, pattern_ids, pattern_table, gate_frozen=False):
            """
            Args:
                h:             (B, T, n_embd) — hidden states
                pattern_ids:   (B, T) int32 — precomputed pattern IDs (-1 = no match)
                pattern_table: nn.Embedding (n_patterns, engram_dim) on same device
                gate_frozen:   bool — if True, force gate=0.5 (don't use learned gate)

            Returns:
                engram_out:    (B, T, n_embd) — additive residual
            """
            B, T, C = h.shape

            # --- 1. Lookup pattern embeddings ---
            # Clamp -1 → 0 for embedding lookup, then zero out no-match positions
            no_match = (pattern_ids < 0)
            safe_ids = pattern_ids.clamp(min=0)
            emb = pattern_table(safe_ids)         # (B, T, engram_dim)
            emb = emb.masked_fill(no_match.unsqueeze(-1), 0.0)

            # --- 2. Project + gate ---
            v = self.W_V(emb)                     # (B, T, n_embd)
            if gate_frozen:
                gate = torch.full((B, T, 1), 0.5, device=h.device, dtype=h.dtype)
            else:
                gate = torch.sigmoid(self.gate_proj(self.ln_gate(h)))  # (B, T, 1)
            out = gate * v

            # --- 3. Stash tensors for diagnostics (computed outside compile) ---
            self._diag_tensors = (gate.detach(), out.detach(), h.detach(), no_match.detach())

            return out

        def collect_diag(self):
            """Call from training loop OUTSIDE torch.compile to avoid graph breaks."""
            tup = getattr(self, "_diag_tensors", None)
            if tup is None:
                return
            gate, out, h, no_match = tup
            self._diag_tensors = None
            with torch.no_grad():
                n_matched = (~no_match).sum().item()
                n_total = no_match.numel()
                self._last_diag = {
                    "gate_mean": gate.mean().item(),
                    "gate_std": gate.std().item(),
                    "output_norm": out.norm(dim=-1).mean().item(),
                    "hidden_norm": h.norm(dim=-1).mean().item(),
                    "hit_rate": n_matched / max(n_total, 1),
                    "matched_gate_mean": gate[~no_match].mean().item() if n_matched > 0 else 0.0,
                    "unmatched_gate_mean": gate[no_match].mean().item() if n_matched < n_total else 0.0,
                }

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

            # Engrams v3: precomputed PMI pattern table (GPU)
            self._engram_map = {}  # schedule_index → engram_module_index
            if config.use_engrams:
                # Pattern embedding table — lives on GPU during training
                self.pattern_table = nn.Embedding(
                    config.engram_table_size, config.engram_dim,
                )
                nn.init.normal_(self.pattern_table.weight, mean=0.0, std=0.01)

                inject_indices = config.engram_inject_indices or (4, 30)
                self.engram_modules = nn.ModuleList([
                    PatternEngramModule(config) for _ in inject_indices
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

            # Engram init: W_V keeps standard init (non-zero for gradient flow).
            # Gate starts slightly closed (bias=-2 → sigmoid(-2)≈0.12).
            if config.use_engrams:
                for em in self.engram_modules:
                    nn.init.normal_(em.gate_proj.weight, mean=0.0, std=0.01)
                    nn.init.constant_(em.gate_proj.bias, -0.5)

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

        def forward(self, idx, targets=None, pattern_ids=None):
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
            use_engrams = self.config.use_engrams and active_map and pattern_ids is not None

            for sched_idx, block in enumerate(self._block_schedule):
                if use_ckpt:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, freqs_cis, use_reentrant=False,
                    )
                else:
                    x = block(x, freqs_cis)

                # Engram injection: AFTER the block, additive residual
                if use_engrams and sched_idx in active_map:
                    mod_idx = active_map[sched_idx]
                    engram_out = self.engram_modules[mod_idx](
                        x, pattern_ids, self.pattern_table,
                        gate_frozen=getattr(self, "_gate_frozen", False),
                    )
                    x = x + engram_out

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

    def __init__(self, data_path: Path, block_size: int, pattern_path: Path = None):
        self.data = np.memmap(str(data_path), dtype=np.uint32, mode="r")
        self.block_size = block_size
        self.n_tokens = len(self.data)
        # Optional: precomputed 5-gram pattern IDs (int32, -1 = no match)
        self.patterns = None
        if pattern_path is not None and pattern_path.exists():
            self.patterns = np.memmap(str(pattern_path), dtype=np.int32, mode="r")
            # pattern_ids covers 5-gram windows: length = n_tokens - 4
            # For token position i, the pattern ending at i is patterns[i - 4]
            # (window starting at i-4 covers tokens [i-4, i-3, i-2, i-1, i])
            logger.info("  Loaded pattern_ids: %d entries from %s", len(self.patterns), pattern_path)

    def __len__(self):
        return self.n_tokens // self.block_size

    def get_batch(self, batch_size: int, device: str = "cuda"):
        import torch
        ix = np.random.randint(0, self.n_tokens - self.block_size - 1, (batch_size,))
        x = np.stack([self.data[i:i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size].astype(np.int64) for i in ix])
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        # Pattern IDs: for each token position, use the 5-gram window ENDING at that position
        # pattern_ids[j] = pattern for window [j, j+1, j+2, j+3, j+4]
        # Token at corpus position (i + t) ends the window starting at (i + t - 4)
        p = None
        if self.patterns is not None:
            n_pat = len(self.patterns)
            p_list = []
            for i in ix:
                pat = np.full(self.block_size, -1, dtype=np.int32)
                # First 4 positions have no complete 5-gram → stay -1
                src_start = max(0, i - 4)
                dst_start = max(0, 4 - i)  # offset into pat if i < 4
                src_end = min(i - 4 + self.block_size, n_pat)
                length = src_end - src_start
                if length > 0:
                    pat[dst_start:dst_start + length] = self.patterns[src_start:src_end]
                p_list.append(pat)
            p = torch.from_numpy(np.stack(p_list)).to(device)

        return x, y, p


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
    # Pattern IDs path: data_dir / "5grams" / "pattern_ids.bin" (if engrams enabled)
    pattern_path = None
    if model_config.use_engrams:
        pattern_path = data_dir / "5grams" / "pattern_ids.bin"
        if not pattern_path.exists():
            logger.warning("Engrams enabled but no pattern_ids.bin at %s — engrams will be disabled", pattern_path)
            pattern_path = None

    train_data = MemmapDataset(data_dir / "train.bin", model_config.block_size, pattern_path=pattern_path)
    # Val: try val pattern_ids, fall back to None (engrams just get -1 = no match during eval)
    val_pattern_path = data_dir / "5grams" / "val_pattern_ids.bin" if pattern_path else None
    if val_pattern_path and not val_pattern_path.exists():
        val_pattern_path = None
    val_data = MemmapDataset(data_dir / "val.bin", model_config.block_size, pattern_path=val_pattern_path)

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
    # Engram v3: pattern table stays on GPU (dense, ~148 MB for 580K × 64)
    if model_config.use_engrams and hasattr(model, "pattern_table"):
        logger.info("Engram v3: pattern table on GPU (%d × %d = %.0f MB)",
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
        model.load_state_dict(checkpoint["model"], strict=False)
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

    # Optimizer
    _raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    param_dict = {pn: p for pn, p in _raw_model.named_parameters() if p.requires_grad}
    # Pattern table gets its own group with LR multiplier, no weight decay
    table_params = []
    decay_params = []
    nodecay_params = []
    for pn, p in param_dict.items():
        if "pattern_table" in pn:
            table_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    # Pattern table: separate LR multiplier, no weight decay, same optimizer
    if table_params:
        optim_groups.append({
            "params": table_params,
            "weight_decay": 0.0,
            "lr": train_config.learning_rate * model_config.engram_table_lr_mult,
        })

    use_fused = (device == "cuda")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=use_fused,
    )

    # Log pattern table info
    table_optimizer = None  # kept for checkpoint compat, no separate optimizer needed
    if model_config.use_engrams and table_params:
        logger.info("Engram pattern table: %d params (%.1f MB), lr_mult=%.1f, in main AdamW",
                     sum(p.numel() for p in table_params),
                     sum(p.numel() for p in table_params) * 4 / 1e6,
                     model_config.engram_table_lr_mult)

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
        for i, param_group in enumerate(optimizer.param_groups):
            if i < 2:
                # decay + nodecay groups
                param_group["lr"] = lr
            else:
                # pattern table group — apply LR multiplier
                param_group["lr"] = lr * model_config.engram_table_lr_mult

        # Engram v3: warmup (full freeze, no eviction needed)
        if model_config.use_engrams:
            engram_active = step >= model_config.engram_warmup_steps
            gate_freeze_end = model_config.engram_warmup_steps + model_config.engram_gate_freeze_steps
            gate_frozen = engram_active and step < gate_freeze_end

            # During warmup: freeze entire engram module + pattern table
            for em in _raw_model.engram_modules.parameters():
                em.requires_grad_(engram_active)
            _raw_model.pattern_table.weight.requires_grad_(engram_active)

            # Freeze gate params during gate-freeze window (table + W_V still learn)
            if gate_frozen:
                for em in _raw_model.engram_modules:
                    em.gate_proj.weight.requires_grad_(False)
                    em.gate_proj.bias.requires_grad_(False)
                    em.ln_gate.weight.requires_grad_(False)

            # Set flag for forward pass
            _raw_model._gate_frozen = gate_frozen

            # Disable engram injection during warmup
            if not engram_active:
                _raw_model._engram_map_active = {}
            else:
                _raw_model._engram_map_active = _raw_model._engram_map

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        if table_optimizer is not None:
            table_optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y, p = train_data.get_batch(train_config.micro_batch_size, device)

            with ctx:
                logits, loss = model(x, y, pattern_ids=p)
                loss = loss / train_config.gradient_accumulation_steps

            if step <= start_step + 1 and micro_step == 0:
                alloc = torch.cuda.memory_allocated() / 1e9
                resv = torch.cuda.memory_reserved() / 1e9
                logger.info("  [MEM] step=%d micro=%d BEFORE backward: alloc=%.1fG reserved=%.1fG",
                            step, micro_step, alloc, resv)
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # (table_optimizer is None in v3 — pattern table is in main AdamW)
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
                    engram_on = step >= model_config.engram_warmup_steps
                    gf_end = model_config.engram_warmup_steps + model_config.engram_gate_freeze_steps
                    log_dict["engram/active"] = 1.0 if engram_on else 0.0
                    log_dict["engram/gate_frozen"] = 1.0 if (engram_on and step < gf_end) else 0.0

                    # Pattern table stats — always logged
                    with torch.no_grad():
                        tbl = _raw_model.pattern_table.weight.data
                        row_norms = tbl.norm(dim=1)
                        log_dict["engram/table_norm_mean"] = row_norms.mean().item()
                        log_dict["engram/table_norm_max"] = row_norms.max().item()
                        log_dict["engram/table_norm_std"] = row_norms.std().item()

                    # Module parameter norms — always logged
                    for mod_idx, em in enumerate(_raw_model.engram_modules):
                        pfx = f"engram/m{mod_idx}"
                        log_dict[f"{pfx}/wv_norm"] = em.W_V.weight.data.norm().item()
                        log_dict[f"{pfx}/gate_bias"] = em.gate_proj.bias.data.item()

                    # Collect diagnostics OUTSIDE compile graph
                    for em in _raw_model.engram_modules:
                        em.collect_diag()
                    # Per-module forward-pass diagnostics (only after warmup)
                    for mod_idx, em in enumerate(_raw_model.engram_modules):
                        diag = getattr(em, "_last_diag", None)
                        if diag is None:
                            continue
                        pfx = f"engram/m{mod_idx}"
                        log_dict[f"{pfx}/gate_mean"] = diag["gate_mean"]
                        log_dict[f"{pfx}/gate_std"] = diag["gate_std"]
                        log_dict[f"{pfx}/output_norm"] = diag["output_norm"]
                        log_dict[f"{pfx}/hidden_norm"] = diag["hidden_norm"]
                        log_dict[f"{pfx}/out_hidden_ratio"] = diag["output_norm"] / max(diag["hidden_norm"], 1e-8)
                        log_dict[f"{pfx}/hit_rate"] = diag["hit_rate"]
                        log_dict[f"{pfx}/matched_gate_mean"] = diag["matched_gate_mean"]
                        log_dict[f"{pfx}/unmatched_gate_mean"] = diag["unmatched_gate_mean"]
                wandb.log(log_dict, step=step)

        # Eval
        if step > 0 and step % train_config.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(train_config.eval_steps):
                    x, y, p = val_data.get_batch(train_config.micro_batch_size, device)
                    with ctx:
                        _, loss = model(x, y, pattern_ids=p)
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
        # Engrams v3 (restored from checkpoint if present)
        use_engrams=cfg.get("use_engrams", False),
        engram_table_size=cfg.get("engram_table_size", 580_262),
        engram_dim=cfg.get("engram_dim", 64),
        engram_inject_indices=tuple(cfg.get("engram_inject_indices", (4, 30))),
    )

    # Build model
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)

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
        # Engrams v3 (PMI pattern memory)
        p.add_argument("--use-engrams", action="store_true", help="Enable PMI pattern engram memory")
        p.add_argument("--engram-table-size", type=int, default=580_262,
                       help="Number of unique patterns (default: 580262 from preprocessing)")
        p.add_argument("--engram-dim", type=int, default=64,
                       help="Per-pattern embedding dimension (default: 64)")
        p.add_argument("--engram-warmup-steps", type=int, default=3000,
                       help="Freeze entire engram module for first N steps (default: 3000)")
        p.add_argument("--engram-gate-freeze-steps", type=int, default=500,
                       help="Force gate open (0.5) for N steps after warmup (default: 500)")
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
        # Engrams v3
        model_config.use_engrams = args.use_engrams
        model_config.engram_table_size = args.engram_table_size
        model_config.engram_dim = args.engram_dim
        model_config.engram_warmup_steps = args.engram_warmup_steps
        model_config.engram_gate_freeze_steps = args.engram_gate_freeze_steps

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
