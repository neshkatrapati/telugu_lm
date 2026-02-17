#!/usr/bin/env python3
"""
Debug: Compare token IDs produced by chat.py vs Space (apply_chat_template).

Run on the remote machine where you have the checkpoint and tokenizer.

Usage:
    python debug_tokenization.py --checkpoint ./sft_checkpoints/best.pt
"""
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", "-t", type=Path, default=Path("./tokenizer"))
    parser.add_argument("--hf-dir", type=str, default="dvitvaai/pothana-chat-300M",
                        help="Path to local HF model dir or Hub model ID")
    args = parser.parse_args()

    # ---- Load our tokenizer (same as chat.py) ----
    from train_tokenizer import MorfessorTokenizer
    our_tokenizer = MorfessorTokenizer(args.tokenizer_dir)

    # Load Morfessor
    import morfessor
    morf_path = Path("./data/morfessor/morfessor_telugu.bin")
    if not morf_path.exists():
        morf_path = Path("./morfessor_telugu.bin")
    morf_io = morfessor.MorfessorIO()
    morf_model = morf_io.read_binary_model_file(str(morf_path))

    # Load HF tokenizer
    from transformers import AutoTokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)

    # Load special tokens from checkpoint
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    special_tokens = checkpoint.get("special_tokens", {})
    print(f"Special tokens: {special_tokens}")
    del checkpoint

    # ---- Segmentation (same as chat.py / Space) ----
    import re
    TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]+")

    def segment_text(text):
        tokens = text.split()
        seg_tokens = []
        for token in tokens:
            if TELUGU_RE.fullmatch(token):
                segments = morf_model.viterbi_segment(token)[0]
                for i, seg in enumerate(segments):
                    if i < len(segments) - 1:
                        seg_tokens.append(seg + "@@")
                    else:
                        seg_tokens.append(seg)
            else:
                seg_tokens.append(token)
        return " ".join(seg_tokens)

    # ---- Test case ----
    system_msg = "మీరు ఒక సహాయకరమైన తెలుగు AI అసిస్టెంట్."
    user_msg = "పోతన గారు ఎవరు ?"

    print("\n" + "=" * 70)
    print("METHOD 1: chat.py style (direct ID construction)")
    print("=" * 70)

    sys_segmented = segment_text(system_msg)
    user_segmented = segment_text(user_msg)
    print(f"  System segmented: '{sys_segmented}'")
    print(f"  User segmented:   '{user_segmented}'")

    # Build IDs the chat.py way
    ids_chatpy = []
    ids_chatpy.append(2)  # <bos>
    ids_chatpy.append(special_tokens["<|system|>"])
    ids_chatpy.extend(our_tokenizer.encode(sys_segmented, add_bos=False, add_eos=False))
    ids_chatpy.append(special_tokens["<|end|>"])
    ids_chatpy.append(special_tokens["<|user|>"])
    ids_chatpy.extend(our_tokenizer.encode(user_segmented, add_bos=False, add_eos=False))
    ids_chatpy.append(special_tokens["<|end|>"])
    ids_chatpy.append(special_tokens["<|assistant|>"])

    print(f"\n  Token IDs ({len(ids_chatpy)} tokens):")
    print(f"  {ids_chatpy}")

    # Decode each token for debugging
    id_to_token = {v: k for k, v in our_tokenizer.token_to_id.items()}
    id_to_special = {v: k for k, v in special_tokens.items()}
    print(f"\n  Token-by-token:")
    for i, tid in enumerate(ids_chatpy):
        if tid in id_to_special:
            name = id_to_special[tid]
        elif tid == 2:
            name = "<bos>"
        elif tid == 3:
            name = "<eos>"
        elif tid in id_to_token:
            name = id_to_token[tid]
        else:
            name = f"<unk:{tid}>"
        print(f"    [{i:3d}] {tid:6d} → '{name}'")

    print("\n" + "=" * 70)
    print("METHOD 2: Space style (apply_chat_template → tokenize)")
    print("=" * 70)

    # Build the prompt the Space way
    messages = [
        {"role": "system", "content": sys_segmented},
        {"role": "user", "content": user_segmented},
    ]
    prompt_str = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n  Chat template output:")
    print(f"  '{prompt_str}'")
    print(f"  (repr: {repr(prompt_str)})")

    # Tokenize
    hf_ids = hf_tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    print(f"\n  Token IDs ({len(hf_ids)} tokens):")
    print(f"  {hf_ids}")

    # Decode each token
    print(f"\n  Token-by-token:")
    for i, tid in enumerate(hf_ids):
        token_str = hf_tokenizer.decode([tid])
        print(f"    [{i:3d}] {tid:6d} → '{token_str}'")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    if ids_chatpy == hf_ids:
        print("\n  ✅ IDENTICAL — both methods produce the same token IDs")
    else:
        print(f"\n  ❌ DIFFERENT!")
        print(f"     chat.py: {len(ids_chatpy)} tokens")
        print(f"     Space:   {len(hf_ids)} tokens")

        # Find first difference
        for i in range(min(len(ids_chatpy), len(hf_ids))):
            if ids_chatpy[i] != hf_ids[i]:
                print(f"\n     First diff at position {i}:")
                print(f"       chat.py: {ids_chatpy[i]}")
                print(f"       Space:   {hf_ids[i]}")

                # Show context
                start = max(0, i - 3)
                end = min(len(ids_chatpy), len(hf_ids), i + 4)
                print(f"\n     chat.py[{start}:{end}]: {ids_chatpy[start:end]}")
                print(f"     Space[{start}:{end}]:   {hf_ids[start:end]}")
                break

        if len(ids_chatpy) != len(hf_ids):
            print(f"\n     Length difference: {len(ids_chatpy)} vs {len(hf_ids)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
