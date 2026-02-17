#!/usr/bin/env python3
"""
Fix tokenizer.json for the SFT model by adding the 4 chat special tokens.

Downloads the existing tokenizer.json from HF, adds the chat tokens,
and saves the corrected version.

Usage:
    python fix_tokenizer_json.py
"""

import json
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "dvitvaai/pothana-chat-300M"
OUTPUT_DIR = Path(__file__).parent

# The 4 SFT special tokens — IDs assigned after base vocab (86071)
SFT_TOKENS = {
    "<|system|>": 86071,
    "<|user|>": 86072,
    "<|assistant|>": 86073,
    "<|end|>": 86074,
}


def main():
    # Download existing tokenizer.json from HF
    print("Downloading tokenizer.json from HF...")
    tok_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")

    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    print(f"Loaded tokenizer.json — current vocab entries: {len(tok_data['model']['vocab'])}")

    # 1. Add tokens to model.vocab
    for token_name, token_id in SFT_TOKENS.items():
        tok_data["model"]["vocab"][token_name] = token_id
        print(f"  Added to vocab: {token_name} -> {token_id}")

    print(f"  New vocab entries: {len(tok_data['model']['vocab'])}")

    # 2. Add to added_tokens list
    for token_name, token_id in SFT_TOKENS.items():
        tok_data["added_tokens"].append({
            "id": token_id,
            "content": token_name,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })
        print(f"  Added to added_tokens: {token_name} (id={token_id})")

    # Save corrected tokenizer.json
    out_path = OUTPUT_DIR / "tokenizer.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved corrected tokenizer.json to {out_path}")
    print(f"  Total vocab: {len(tok_data['model']['vocab'])}")
    print(f"  Total added_tokens: {len(tok_data['added_tokens'])}")


if __name__ == "__main__":
    main()
