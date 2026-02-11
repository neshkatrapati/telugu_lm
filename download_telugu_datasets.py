#!/usr/bin/env python3
"""
Telugu Language Model Training Data Downloader
===============================================
Downloads AI4Bharat Telugu datasets for LLM pre-training:
  1. Sangraha (Verified, Unverified, Synthetic) — ~16.3B tokens
  2. IndicCorp v2 — ~731M tokens

Total Telugu data: ~16.3B+ tokens (~35+ GB on disk)

Requirements:
    pip install datasets huggingface_hub tqdm

Usage:
    python download_telugu_datasets.py                    # Download all datasets
    python download_telugu_datasets.py --dataset sangraha # Only Sangraha
    python download_telugu_datasets.py --dataset indiccorp # Only IndicCorp v2
    python download_telugu_datasets.py --subset verified   # Only Sangraha verified split
    python download_telugu_datasets.py --streaming         # Stream & save (low memory)
    python download_telugu_datasets.py --output /mnt/data   # Custom output directory
    python download_telugu_datasets.py --format jsonl      # Output as JSONL (default: parquet)
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")

    if missing:
        logger.error(
            "Missing required packages: %s\n"
            "Install them with:\n"
            "    pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
SANGRAHA_SUBSETS = {
    "verified": {
        "data_dir": "verified/tel",
        "description": "Human-verified web & PDF sources (~3.7B tokens, ~15.1 GB)",
        "approx_size_gb": 15.1,
        "approx_tokens": "3.7B",
    },
    "unverified": {
        "data_dir": "unverified/tel",
        "description": "Filtered CulturaX & MADLAD-400 (~647M tokens, ~2.4 GB)",
        "approx_size_gb": 2.4,
        "approx_tokens": "647M",
    },
    "synthetic": {
        "data_dir": "synthetic/tel_Telu",
        "description": "English Wikipedia translated to Telugu (~11.9B tokens, ~18.1 GB)",
        "approx_size_gb": 18.1,
        "approx_tokens": "11.9B",
    },
    "synthetic_romanized": {
        "data_dir": "synthetic/tel_Latn",
        "description": "Romanized/Latin-script version of synthetic data",
        "approx_size_gb": 10.0,
        "approx_tokens": "~11.9B (romanized)",
    },
}

INDICCORP_CONFIG = {
    "repo": "ai4bharat/IndicCorpV2",
    "config_name": "indiccorp_v2",
    "data_dir": "data/tel_Telu",
    "description": "IndicCorp v2 Telugu (~731M tokens, ~15.8 GB .txt)",
    "approx_size_gb": 15.8,
    "approx_tokens": "731M",
}


# ---------------------------------------------------------------------------
# Disk space check
# ---------------------------------------------------------------------------
def check_disk_space(output_dir: Path, required_gb: float):
    """Warn if disk space might be insufficient."""
    try:
        stat = os.statvfs(str(output_dir))
        available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if available_gb < required_gb * 1.5:  # 1.5x buffer for caching
            logger.warning(
                "Low disk space! Available: %.1f GB, Required: ~%.1f GB "
                "(+ HuggingFace cache). Consider using --streaming mode.",
                available_gb,
                required_gb,
            )
            return False
        logger.info("Disk space OK: %.1f GB available (need ~%.1f GB)", available_gb, required_gb)
        return True
    except (OSError, AttributeError):
        logger.warning("Could not check disk space. Ensure you have enough room.")
        return True


# ---------------------------------------------------------------------------
# Already-downloaded check
# ---------------------------------------------------------------------------
MINIMUM_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB — anything smaller is likely corrupt/incomplete


def find_existing_file(output_path: Path, name: str) -> Path | None:
    """Check if a dataset file already exists in any supported format."""
    for ext in (".parquet", ".jsonl", ".txt"):
        candidate = output_path / f"{name}{ext}"
        if candidate.exists() and candidate.stat().st_size > MINIMUM_FILE_SIZE_BYTES:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_dataset(dataset, output_path: Path, fmt: str, name: str):
    """Save a HuggingFace dataset to disk in the chosen format."""
    output_path.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        filepath = output_path / f"{name}.parquet"
        logger.info("Saving %s as Parquet -> %s", name, filepath)
        dataset.to_parquet(str(filepath))
    elif fmt == "jsonl":
        filepath = output_path / f"{name}.jsonl"
        logger.info("Saving %s as JSONL -> %s", name, filepath)
        dataset.to_json(str(filepath), lines=True)
    elif fmt == "txt":
        filepath = output_path / f"{name}.txt"
        logger.info("Saving %s as plain text -> %s", name, filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            for row in dataset:
                f.write(row["text"] + "\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")

    size_gb = filepath.stat().st_size / (1024 ** 3)
    logger.info("Saved %s (%.2f GB)", filepath.name, size_gb)


def save_streaming_dataset(dataset_iter, output_path: Path, fmt: str, name: str):
    """Stream a dataset and write incrementally (low memory usage)."""
    from tqdm import tqdm

    output_path.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        filepath = output_path / f"{name}.txt"
    elif fmt == "jsonl":
        filepath = output_path / f"{name}.jsonl"
    else:
        filepath = output_path / f"{name}.jsonl"
        logger.info("Streaming mode uses JSONL format (overriding '%s')", fmt)
        fmt = "jsonl"

    logger.info("Streaming %s -> %s", name, filepath)
    import json

    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for row in tqdm(dataset_iter, desc=f"Streaming {name}", unit=" docs"):
            if fmt == "txt":
                f.write(row["text"] + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    size_gb = filepath.stat().st_size / (1024 ** 3)
    logger.info("Saved %s -- %d documents (%.2f GB)", filepath.name, count, size_gb)


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------
def download_sangraha(subsets, output_dir: Path, fmt: str, streaming: bool):
    """Download AI4Bharat Sangraha Telugu data."""
    from datasets import load_dataset

    for subset_name in subsets:
        if subset_name not in SANGRAHA_SUBSETS:
            logger.warning("Unknown Sangraha subset '%s', skipping.", subset_name)
            continue

        config = SANGRAHA_SUBSETS[subset_name]
        out_path = output_dir / "sangraha"
        file_name = f"telugu_{subset_name}"

        # Skip if already downloaded
        existing = find_existing_file(out_path, file_name)
        if existing:
            size_gb = existing.stat().st_size / (1024 ** 3)
            logger.info("=" * 70)
            logger.info(
                "SKIPPING Sangraha [%s] — already exists: %s (%.2f GB)",
                subset_name, existing, size_gb,
            )
            logger.info("=" * 70)
            continue

        logger.info("=" * 70)
        logger.info("Downloading Sangraha [%s]", subset_name)
        logger.info("   %s", config["description"])
        logger.info("=" * 70)

        start = time.time()

        try:
            if streaming:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    data_dir=config["data_dir"],
                    streaming=True,
                    split="train",
                )
                save_streaming_dataset(ds, out_path, fmt, file_name)
            else:
                ds = load_dataset(
                    "ai4bharat/sangraha",
                    data_dir=config["data_dir"],
                    split="train",
                )
                logger.info(
                    "Loaded %d documents from Sangraha [%s]",
                    len(ds),
                    subset_name,
                )
                save_dataset(ds, out_path, fmt, file_name)

            elapsed = time.time() - start
            logger.info("Sangraha [%s] completed in %.1f minutes", subset_name, elapsed / 60)

        except Exception as e:
            logger.error("Failed to download Sangraha [%s]: %s", subset_name, e)
            logger.info("Try running with --streaming if you're running out of memory.")
            raise


def download_indiccorp(output_dir: Path, fmt: str, streaming: bool):
    """Download AI4Bharat IndicCorp v2 Telugu data."""
    from datasets import load_dataset

    out_path = output_dir / "indiccorp_v2"
    file_name = "telugu_indiccorp_v2"

    # Skip if already downloaded
    existing = find_existing_file(out_path, file_name)
    if existing:
        size_gb = existing.stat().st_size / (1024 ** 3)
        logger.info("=" * 70)
        logger.info(
            "SKIPPING IndicCorp v2 — already exists: %s (%.2f GB)",
            existing, size_gb,
        )
        logger.info("=" * 70)
        return

    logger.info("=" * 70)
    logger.info("Downloading IndicCorp v2 [Telugu]")
    logger.info("   %s", INDICCORP_CONFIG["description"])
    logger.info("=" * 70)

    start = time.time()

    try:
        if streaming:
            ds = load_dataset(
                INDICCORP_CONFIG["repo"],
                INDICCORP_CONFIG["config_name"],
                data_dir=INDICCORP_CONFIG["data_dir"],
                streaming=True,
                split="train",
            )
            save_streaming_dataset(ds, out_path, fmt, file_name)
        else:
            ds = load_dataset(
                INDICCORP_CONFIG["repo"],
                INDICCORP_CONFIG["config_name"],
                data_dir=INDICCORP_CONFIG["data_dir"],
                split="train",
            )
            logger.info("Loaded %d documents from IndicCorp v2", len(ds))
            save_dataset(ds, out_path, fmt, file_name)

        elapsed = time.time() - start
        logger.info("IndicCorp v2 completed in %.1f minutes", elapsed / 60)

    except Exception as e:
        logger.error("Failed to download IndicCorp v2: %s", e)
        logger.info("Try running with --streaming if you're running out of memory.")
        raise


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(output_dir: Path):
    """Print summary of all downloaded files."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)

    total_size = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_size += size
            size_gb = size / (1024 ** 3)
            logger.info("  %-50s  %.2f GB", str(path.relative_to(output_dir)), size_gb)

    logger.info("-" * 70)
    logger.info("  Total: %.2f GB", total_size / (1024 ** 3))
    logger.info("  Location: %s", output_dir.resolve())
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download AI4Bharat Telugu datasets for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Download everything
  %(prog)s --dataset sangraha               # Only Sangraha (all subsets)
  %(prog)s --dataset indiccorp              # Only IndicCorp v2
  %(prog)s --subset verified                # Only Sangraha verified split
  %(prog)s --subset verified unverified     # Verified + Unverified splits
  %(prog)s --streaming                      # Low-memory streaming mode
  %(prog)s --streaming --format txt         # Stream as plain text files
  %(prog)s --output ./telugu_data           # Custom output directory
  %(prog)s --format jsonl                   # Save as JSONL files
  %(prog)s --no-synthetic                   # Skip synthetic (translated) data
        """,
    )

    parser.add_argument(
        "--dataset",
        choices=["all", "sangraha", "indiccorp"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=["verified", "unverified", "synthetic", "synthetic_romanized"],
        default=None,
        help="Sangraha subsets to download (default: verified, unverified, synthetic)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic (machine-translated) data from Sangraha",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory (default: ./data in current working directory)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "jsonl", "txt"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (low memory, writes incrementally)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (if datasets are gated). Usually not needed.",
    )

    args = parser.parse_args()

    # Check dependencies first
    check_dependencies()

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which Sangraha subsets to download
    if args.subset:
        sangraha_subsets = args.subset
    elif args.no_synthetic:
        sangraha_subsets = ["verified", "unverified"]
    else:
        sangraha_subsets = ["verified", "unverified", "synthetic"]

    # Estimate total download size
    total_gb = 0.0
    if args.dataset in ("all", "sangraha"):
        for s in sangraha_subsets:
            total_gb += SANGRAHA_SUBSETS.get(s, {}).get("approx_size_gb", 0)
    if args.dataset in ("all", "indiccorp"):
        total_gb += INDICCORP_CONFIG["approx_size_gb"]

    # Print plan
    logger.info("=" * 70)
    logger.info("Telugu LM Data Downloader")
    logger.info("=" * 70)
    logger.info("  Dataset(s):      %s", args.dataset)
    if args.dataset in ("all", "sangraha"):
        logger.info("  Sangraha splits: %s", ", ".join(sangraha_subsets))
    logger.info("  Output format:   %s", args.format)
    logger.info("  Output dir:      %s", output_dir.resolve())
    logger.info("  Streaming:       %s", "Yes" if args.streaming else "No")
    logger.info("  Est. download:   ~%.1f GB", total_gb)
    logger.info("=" * 70)

    # Check disk space
    check_disk_space(output_dir, total_gb)

    # Download
    start_total = time.time()

    if args.dataset in ("all", "sangraha"):
        download_sangraha(sangraha_subsets, output_dir, args.format, args.streaming)

    if args.dataset in ("all", "indiccorp"):
        download_indiccorp(output_dir, args.format, args.streaming)

    elapsed_total = time.time() - start_total

    # Summary
    print_summary(output_dir)
    logger.info("All downloads completed in %.1f minutes!", elapsed_total / 60)


if __name__ == "__main__":
    main()
