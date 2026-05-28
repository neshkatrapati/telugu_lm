#!/usr/bin/env python3
"""
Stage A data prep: chunk corpus → embed → FAISS → neighbor lookup.

Produces (under --out-dir):
  meta.json           run metadata
  chunk_offsets.npy   int64 (N,) — starting position in train.bin for each chunk
  embeddings.fp16     (N, 1024) — multilingual-e5-large embeddings
  faiss.index         FAISS HNSW index (cosine)
  neighbors.npy       int32 (N, K) — neighbor chunk IDs; -1 = filtered
  neighbor_sims.fp16  (N, K) — corresponding cosine similarities

Usage:
  python build_retrieval_pretrain_data.py \
      --train-bin /workspace/telugu_lm/train-data-v2/train.bin \
      --tokenizer /workspace/telugu_lm/hf_release/pothana-base-v2-225M \
      --out-dir /workspace/telugu_lm/stage_a_data \
      --n-chunks 1000000 \
      --chunk-size 512
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


SPECIAL_TOKEN_STRS = {"<pad>", "<unk>", "<bos>", "<eos>",
                       "<search>", "</search>", "<retrieved>", "</retrieved>",
                       "<doc>", "</doc>", "<cite>", "<think>", "</think>"}


def detokenize_chunks(chunks_uint32: np.ndarray, id_to_token: dict):
    """Decode each row of (N, chunk_size) uint32 IDs to text using a direct
    id→token dict + @@ stripping. ~50x faster than HF batch_decode.
    """
    texts = []
    for row in chunks_uint32:
        # Lookup tokens, skip special tokens
        parts = []
        for tid in row:
            tok = id_to_token.get(int(tid))
            if tok is None or tok in SPECIAL_TOKEN_STRS:
                continue
            parts.append(tok)
        text = " ".join(parts)
        # @@ is a continuation-prefix marker: "మా @@కు" → "మాకు"
        text = text.replace(" @@", "").replace("@@", "")
        texts.append(text.strip())
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True,
                    help="HF tokenizer dir (e.g. hf_release/pothana-base-v2-225M)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--n-chunks", type=int, default=1_000_000,
                    help="Number of chunks to sample uniformly across the file")
    ap.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-large")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sim-threshold", type=float, default=0.9,
                    help="Drop neighbors with cosine sim above this (likely near-dups)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Embedding batch size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Sample chunk offsets (stride + jitter — fast for huge N) =====
    print(f"[1/5] sampling chunk offsets from {args.train_bin}", flush=True)
    t0 = time.time()
    n_tokens = args.train_bin.stat().st_size // 4  # uint32
    print(f"      train.bin: {n_tokens:,} tokens", flush=True)
    rng = np.random.RandomState(args.seed)
    max_start = n_tokens - args.chunk_size - 1
    stride = max_start // args.n_chunks
    # Each chunk falls in its own stride-sized window; random offset within window
    base = (np.arange(args.n_chunks, dtype=np.int64)) * stride
    jitter = rng.randint(0, max(1, stride), size=args.n_chunks).astype(np.int64)
    chunk_offsets = (base + jitter).astype(np.int64)
    np.save(args.out_dir / "chunk_offsets.npy", chunk_offsets)
    print(f"      sampled {len(chunk_offsets):,} chunks  ({time.time()-t0:.1f}s, stride={stride})", flush=True)

    # ===== Step 2: Load vocab (direct id→token map) + e5 model =====
    print(f"[2/5] loading vocab + e5 model on {args.device}", flush=True)
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    # Load id→token map directly from the SOURCE tokenizer JSON (custom morfessor_bpe_telugu_v4 format)
    # — we look for it under tokenizer-new-v5 or fall back to the HF dir's tokenizer.json
    src_tok_path = Path("/workspace/telugu_lm/tokenizer-new-v5/tokenizer.json")
    if src_tok_path.exists():
        src_data = json.load(open(src_tok_path))
        id_to_token = {v: k for k, v in src_data["token_to_id"].items()}
        vocab_size = len(id_to_token)
    else:
        # Fall back: load HF tokenizer.json (WordLevel) and invert
        hf_tok_data = json.load(open(args.tokenizer / "tokenizer.json"))
        vocab_dict = hf_tok_data.get("model", {}).get("vocab", {})
        id_to_token = {v: k for k, v in vocab_dict.items()}
        vocab_size = len(id_to_token)

    embedder = SentenceTransformer(args.embedding_model, device=args.device)
    embedder.eval()
    if args.device.startswith("cuda"):
        embedder.half()
    emb_dim = embedder.get_sentence_embedding_dimension()
    print(f"      vocab={vocab_size}, e5 dim={emb_dim}  ({time.time()-t0:.1f}s)", flush=True)

    # ===== Step 3: Embed all chunks =====
    print(f"[3/5] embedding {args.n_chunks:,} chunks (batch={args.batch_size})", flush=True)
    train_mem = np.memmap(str(args.train_bin), dtype=np.uint32, mode="r")
    embeddings = np.zeros((args.n_chunks, emb_dim), dtype=np.float16)

    # Process in groups of N batches to amortize tokenizer cost
    GROUP = args.batch_size * 8  # decode this many at once
    t0 = time.time()
    last_log = t0
    for grp_start in range(0, args.n_chunks, GROUP):
        grp_end = min(grp_start + GROUP, args.n_chunks)
        # Read chunks
        rows = np.zeros((grp_end - grp_start, args.chunk_size), dtype=np.uint32)
        for i, off in enumerate(chunk_offsets[grp_start:grp_end]):
            rows[i] = train_mem[off : off + args.chunk_size]
        # Decode to text
        texts = detokenize_chunks(rows, id_to_token)
        # Prefix with "passage: " per e5 convention (https://huggingface.co/intfloat/multilingual-e5-large)
        texts = [f"passage: {t}" for t in texts]
        # Embed
        with torch.no_grad():
            vecs = embedder.encode(
                texts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine via L2-normalized dot product
                show_progress_bar=False,
            )
        embeddings[grp_start:grp_end] = vecs.astype(np.float16)
        # Log every ~30 sec
        if time.time() - last_log >= 30:
            elapsed = time.time() - t0
            pct = 100 * grp_end / args.n_chunks
            rate = grp_end / elapsed
            eta = (args.n_chunks - grp_end) / rate / 60
            print(f"      [{pct:5.1f}%] {grp_end:>9,}/{args.n_chunks:,}  {rate:>6.0f} chunks/s  ETA {eta:5.1f}m", flush=True)
            last_log = time.time()
    embed_time = time.time() - t0
    print(f"      done in {embed_time/60:.1f} min  ({args.n_chunks/embed_time:.0f} chunks/s)")
    np.save(args.out_dir / "embeddings.fp16.npy", embeddings)
    print(f"      saved embeddings.fp16.npy ({embeddings.nbytes/1e9:.2f} GB)")

    # ===== Step 4: Build FAISS index =====
    print(f"[4/5] building FAISS HNSW index", flush=True)
    import faiss
    t0 = time.time()
    # HNSW with cosine similarity (use inner product on normalized vectors)
    index = faiss.IndexHNSWFlat(emb_dim, 32)  # M=32 (default for quality)
    index.hnsw.efConstruction = 100
    # IndexHNSWFlat doesn't have a native cosine metric; use inner product (METRIC_INNER_PRODUCT)
    # but IndexHNSWFlat uses L2 by default. Since our vectors are normalized, L2² = 2 - 2·cos.
    # We rank by L2 distance (ascending), equivalent to ranking by cos sim (descending).
    # We'll convert L2 distance back to cosine sim at the end.
    index.add(embeddings.astype(np.float32))
    print(f"      index built in {time.time()-t0:.1f}s, ntotal={index.ntotal}")

    # Save the index (may be skipped — we have neighbors and embeddings cached)
    faiss.write_index(index, str(args.out_dir / "faiss.index"))

    # ===== Step 5: Search top-K neighbors and filter =====
    print(f"[5/5] searching top-{args.top_k+1} neighbors and filtering", flush=True)
    t0 = time.time()
    # Search for K+1 (we'll drop self)
    SEARCH_K = args.top_k + 1
    BATCH = 2048  # query batch
    neighbors = np.full((args.n_chunks, args.top_k), -1, dtype=np.int32)
    neighbor_sims = np.zeros((args.n_chunks, args.top_k), dtype=np.float16)
    index.hnsw.efSearch = 64

    for start in range(0, args.n_chunks, BATCH):
        end = min(start + BATCH, args.n_chunks)
        D, I = index.search(embeddings[start:end].astype(np.float32), SEARCH_K)
        # D is L2² distance for normalized vecs: D = 2 - 2·cos → cos = 1 - D/2
        cos_sims = 1.0 - D / 2.0  # shape (B, SEARCH_K)
        for i in range(end - start):
            row_ids = I[i]
            row_sims = cos_sims[i]
            kept = []
            for nid, sim in zip(row_ids, row_sims):
                if nid == start + i:
                    continue  # self
                if sim > args.sim_threshold:
                    continue  # near-duplicate
                kept.append((nid, sim))
                if len(kept) >= args.top_k:
                    break
            for k, (nid, sim) in enumerate(kept):
                neighbors[start + i, k] = nid
                neighbor_sims[start + i, k] = sim
        if (end // BATCH) % 50 == 0:
            print(f"      searched {end:,}/{args.n_chunks:,}  ({(time.time()-t0)/60:.1f}m elapsed)", flush=True)
    print(f"      done in {(time.time()-t0)/60:.1f} min")
    np.save(args.out_dir / "neighbors.npy", neighbors)
    np.save(args.out_dir / "neighbor_sims.fp16.npy", neighbor_sims)

    # Stats
    n_valid = (neighbors >= 0).sum(axis=1)
    print(f"      neighbors per chunk: mean {n_valid.mean():.2f}  median {int(np.median(n_valid))}  min {n_valid.min()}  max {n_valid.max()}")
    print(f"      chunks with <3 valid neighbors: {(n_valid < 3).sum():,}")
    print(f"      mean of top-1 sims (valid): {neighbor_sims[neighbors >= 0].mean():.4f}")

    # ===== Write meta =====
    meta = {
        "train_bin": str(args.train_bin),
        "tokenizer": str(args.tokenizer),
        "embedding_model": args.embedding_model,
        "n_chunks": args.n_chunks,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "sim_threshold": args.sim_threshold,
        "embed_dim": emb_dim,
        "seed": args.seed,
        "embed_time_min": embed_time / 60,
        "stats": {
            "valid_neighbors_per_chunk_mean": float(n_valid.mean()),
            "valid_neighbors_per_chunk_median": int(np.median(n_valid)),
            "chunks_lt_3_neighbors": int((n_valid < 3).sum()),
            "mean_top1_sim": float(neighbor_sims[neighbors >= 0].mean()),
        },
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Output: {args.out_dir}")
    for p in sorted(args.out_dir.iterdir()):
        sz = p.stat().st_size
        unit = "GB" if sz > 1e9 else "MB"
        scaled = sz / 1e9 if sz > 1e9 else sz / 1e6
        print(f"  {p.name}  ({scaled:.2f} {unit})")


if __name__ == "__main__":
    main()
