#!/usr/bin/env python3
"""
Upload corrected SFT model card and config files to HuggingFace.

This fixes the pothana-chat-300M repo which was uploaded with base model
configs instead of SFT configs.

Files updated:
  - README.md         (SFT model card with chat template docs)
  - config.json       (eos_token_id includes <|end|>)
  - generation_config.json  (SFT defaults: temp=0.7, max=256)
  - special_tokens_map.json (includes chat special tokens)
  - tokenizer_config.json   (includes chat_template jinja2)
  - tokenizer.json          (includes 4 chat token vocab entries)

Usage:
    # First login:
    python -c "from huggingface_hub import login; login()"

    # Then upload:
    python upload_fixes.py
"""

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "dvitvaai/pothana-chat-300M"
FIX_DIR = Path(__file__).parent

FILES_TO_UPLOAD = [
    "README.md",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def main():
    api = HfApi()

    print(f"Uploading fixes to {REPO_ID}...")
    print()

    for filename in FILES_TO_UPLOAD:
        filepath = FIX_DIR / filename
        if not filepath.exists():
            print(f"  SKIP {filename} (not found)")
            continue

        print(f"  Uploading {filename}...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=REPO_ID,
            commit_message=f"Fix {filename}: update to SFT chat model config",
        )
        print("OK")

    print()
    print(f"All fixes uploaded to https://huggingface.co/{REPO_ID}")
    print()
    print("Changes:")
    print("  - README.md: SFT model card with chat template docs, usage examples")
    print("  - config.json: eos_token_id now includes <|end|> (86074)")
    print("  - generation_config.json: SFT defaults (temp=0.7, max_new_tokens=256)")
    print("  - special_tokens_map.json: added chat special tokens")
    print("  - tokenizer_config.json: added chat_template (Jinja2)")
    print("  - tokenizer.json: added <|system|>, <|user|>, <|assistant|>, <|end|> to vocab")


if __name__ == "__main__":
    main()
