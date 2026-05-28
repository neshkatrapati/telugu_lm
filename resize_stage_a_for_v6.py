#!/usr/bin/env python3
"""
Resize Stage A checkpoint's embedding and lm_head from 47,831 → new_vocab_size,
initializing new token rows via "smart init" — each new token's embedding is
the mean of its old BPE-decomposition's embeddings.

This gives the model a sensible starting point: new tokens begin training with
a representation similar to how they were composed before, then Stage A+
fine-tunes them.

Usage:
    python resize_stage_a_for_v6.py \\
        --in-ckpt checkpoints/stage-a/final.pt \\
        --old-tok tokenizer-new-v5 \\
        --new-tok tokenizer-new-v6 \\
        --out-ckpt checkpoints/stage-a/final-v6init.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ckpt", type=Path, required=True)
    ap.add_argument("--old-tok", type=Path, required=True,
                    help="Tokenizer dir with vocab.txt + BPE merges (MorfessorTokenizer compatible)")
    ap.add_argument("--new-tok", type=Path, required=True)
    ap.add_argument("--out-ckpt", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, "/workspace/telugu_lm")
    from train_tokenizer import MorfessorTokenizer

    # Load old tokenizer for encoding new word forms
    print(f"[1/4] loading old tokenizer from {args.old_tok}", flush=True)
    old_tok = MorfessorTokenizer(args.old_tok)

    # Load new vocab to know what tokens were added
    print(f"[2/4] loading new vocab from {args.new_tok}", flush=True)
    new_tok_json = json.load(open(args.new_tok / "tokenizer.json"))
    new_vocab_size = new_tok_json["vocab_size"]

    added_rec = json.load(open(args.new_tok / "added_tokens_v6.json"))
    added = added_rec["added"]
    old_vocab_size = added_rec["old_vocab_size"]
    print(f"   old vocab: {old_vocab_size:,}")
    print(f"   new vocab: {new_vocab_size:,}")
    print(f"   added:     {len(added):,}")

    # Load checkpoint
    print(f"[3/4] loading checkpoint {args.in_ckpt}", flush=True)
    ckpt = torch.load(str(args.in_ckpt), map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt["model"]
    cfg = ckpt["config"]
    print(f"   step: {ckpt.get('step')}  val_loss: {ckpt.get('val_loss')}  best: {ckpt.get('best_val_loss')}")

    # Find embedding + lm_head
    wte_key = "transformer.wte.weight"
    lm_head_key = "lm_head.weight"
    if wte_key not in sd:
        raise SystemExit(f"missing {wte_key} in checkpoint")
    if lm_head_key not in sd:
        raise SystemExit(f"missing {lm_head_key} in checkpoint")
    wte_old = sd[wte_key]
    lm_head_old = sd[lm_head_key]
    print(f"   wte shape:     {tuple(wte_old.shape)}")
    print(f"   lm_head shape: {tuple(lm_head_old.shape)}")

    n_old, dim = wte_old.shape
    if n_old != old_vocab_size:
        raise SystemExit(f"wte rows={n_old} != old_vocab_size={old_vocab_size}")

    # Build new tensors
    print(f"[4/4] computing smart-init for {len(added):,} new tokens", flush=True)
    wte_new = torch.zeros((new_vocab_size, dim), dtype=wte_old.dtype)
    lm_new  = torch.zeros((new_vocab_size, dim), dtype=lm_head_old.dtype)
    wte_new[:n_old] = wte_old
    lm_new[:n_old]  = lm_head_old

    # For each added word, encode via OLD tokenizer's BPE fallback → mean embeddings
    n_smart = 0
    n_fallback_random = 0
    t0 = time.time()
    for a in added:
        word = a["word"]
        new_id = a["id"]
        # OLD vocab lookup
        tid = old_tok.token_to_id.get(word)
        if tid is not None:
            # Word was already in old vocab (shouldn't happen for added tokens but defensive)
            wte_new[new_id] = wte_old[tid]
            lm_new[new_id]  = lm_head_old[tid]
            n_smart += 1
            continue
        # BPE fallback
        try:
            sub_ids = old_tok._encode_token_bpe(word)
        except Exception:
            sub_ids = []
        if sub_ids:
            sub_ids = torch.tensor(sub_ids, dtype=torch.long)
            wte_new[new_id] = wte_old.index_select(0, sub_ids).mean(dim=0).to(wte_old.dtype)
            lm_new[new_id]  = lm_head_old.index_select(0, sub_ids).mean(dim=0).to(lm_head_old.dtype)
            n_smart += 1
        else:
            # Last resort: small random init
            wte_new[new_id] = torch.randn(dim, dtype=wte_old.dtype) * 0.02
            lm_new[new_id]  = torch.randn(dim, dtype=lm_head_old.dtype) * 0.02
            n_fallback_random += 1

    print(f"   smart-init: {n_smart:,}  random fallback: {n_fallback_random}  ({time.time()-t0:.1f}s)")

    # Replace tensors
    sd[wte_key] = wte_new
    sd[lm_head_key] = lm_new

    # Update vocab_size in config
    cfg["vocab_size"] = new_vocab_size

    # Save
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out = dict(ckpt)
    out["model"] = sd
    out["config"] = cfg
    out["vocab_resize"] = {
        "from": old_vocab_size,
        "to": new_vocab_size,
        "new_tokens": len(added),
        "smart_init": n_smart,
        "random_fallback": n_fallback_random,
        "source_ckpt": str(args.in_ckpt),
    }
    torch.save(out, str(args.out_ckpt))
    sz = args.out_ckpt.stat().st_size / 1e9
    print(f"\nsaved {args.out_ckpt} ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
