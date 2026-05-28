#!/usr/bin/env python3
"""
Extract 5-gram counts from a corpus using KenLM's count_ngrams,
then decode the binary output to TSV + DuckDB.

Usage:
    python kenlm_5gram_counts.py <corpus_file> [options]

    e.g. python kenlm_5gram_counts.py ukwac_1M_san.txt
         python kenlm_5gram_counts.py corpus.txt --kenlm-bin /usr/local/bin
         python kenlm_5gram_counts.py corpus.txt --order 5 --min-count 2 --out-dir output/

Requires: KenLM count_ngrams binary
"""

import sys
import os
import struct
import argparse
import subprocess
import tempfile
import time
from tqdm import tqdm


def find_count_ngrams(kenlm_bin_dir=None):
    """Find the count_ngrams binary."""
    candidates = []
    if kenlm_bin_dir:
        candidates.append(os.path.join(kenlm_bin_dir, "count_ngrams"))

    # Common locations
    candidates += [
        "count_ngrams",  # on PATH
        os.path.expanduser("~/kenlm/build/bin/count_ngrams"),
        "/usr/local/bin/count_ngrams",
    ]

    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        # Also try via `which`
        try:
            result = subprocess.run(["which", c], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

    return None


def run_count_ngrams(corpus_path, order, kenlm_bin_dir, memory, tmp_dir):
    """Run KenLM count_ngrams and return paths to binary + vocab files."""
    binary = find_count_ngrams(kenlm_bin_dir)
    if binary is None:
        print("ERROR: count_ngrams not found.")
        print("  Specify --kenlm-bin /path/to/kenlm/build/bin/")
        sys.exit(1)

    print(f"Using: {binary}")

    vocab_path = os.path.join(tmp_dir, "vocab.list")
    counts_path = os.path.join(tmp_dir, "ngram_counts.bin")

    cmd = [
        binary,
        "-o", str(order),
        "--write_vocab_list", vocab_path,
        "-T", tmp_dir,
        "-S", memory,
    ]

    print(f"Running: {' '.join(cmd)} < {corpus_path}")
    print(f"  (this may take a while for large corpora)\n")

    t0 = time.time()
    with open(corpus_path, "r") as fin, open(counts_path, "wb") as fout:
        proc = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        print(f"ERROR: count_ngrams failed:\n{proc.stderr.decode()}")
        sys.exit(1)

    elapsed = time.time() - t0
    bin_size = os.path.getsize(counts_path) / 1e6
    print(f"  count_ngrams finished in {elapsed:.1f}s")
    print(f"  Binary counts: {bin_size:.1f} MB")
    print(f"  Vocab list: {os.path.getsize(vocab_path) / 1e6:.1f} MB\n")

    return counts_path, vocab_path


def load_vocab(vocab_path):
    """Load null-delimited vocab list."""
    with open(vocab_path, "rb") as f:
        data = f.read()
    words = data.split(b"\x00")
    if words and words[-1] == b"":
        words = words[:-1]
    vocab = {i: w.decode("utf-8", errors="replace") for i, w in enumerate(words)}
    print(f"Vocab loaded: {len(vocab):,} entries")
    return vocab


def decode_binary(counts_path, vocab, order, min_count, tsv_path):
    """Decode binary count file to TSV."""
    # Record: order × uint32 (vocab IDs) + uint64 (count)
    record_size = order * 4 + 8
    file_size = os.path.getsize(counts_path)
    num_records = file_size // record_size
    remainder = file_size % record_size

    if remainder != 0:
        print(f"WARNING: file size {file_size} not divisible by record size "
              f"{record_size} (remainder={remainder})")

    print(f"Decoding {num_records:,} {order}-gram records...")

    id_fmt = f"<{order}I"
    id_size = order * 4
    written = 0
    t0 = time.time()

    header = "\t".join(f"w{i+1}" for i in range(order)) + "\tcount\n"

    with open(counts_path, "rb") as fin, \
         open(tsv_path, "w", encoding="utf-8") as fout:
        fout.write(header)

        for _ in tqdm(range(num_records), desc="Decoding", unit="ngram",
                      unit_scale=True):
            rec = fin.read(record_size)
            ids = struct.unpack(id_fmt, rec[:id_size])
            count = struct.unpack("<Q", rec[id_size:id_size + 8])[0]

            if count < min_count:
                continue

            words = [vocab.get(id, f"<UNK:{id}>") for id in ids]
            fout.write("\t".join(words) + f"\t{count}\n")
            written += 1

    elapsed = time.time() - t0
    tsv_size = os.path.getsize(tsv_path) / 1e6
    print(f"  {written:,} {order}-grams written ({num_records - written:,} "
          f"filtered out) in {elapsed:.1f}s")
    print(f"  TSV: {tsv_size:.1f} MB")
    return written


def load_duckdb(tsv_path, db_path, order):
    """Load TSV into DuckDB."""
    try:
        import duckdb
    except ImportError:
        print("\n  duckdb not installed, skipping DB output.")
        print("  Install with: pip install duckdb")
        return

    print(f"\nLoading into DuckDB at {db_path}...")
    t0 = time.time()
    db = duckdb.connect(db_path)

    cols_spec = ", ".join(f"'w{i+1}': 'VARCHAR'" for i in range(order))
    db.execute("DROP TABLE IF EXISTS fivegrams")
    db.execute(f"""
        CREATE TABLE fivegrams AS
        SELECT * FROM read_csv(
            '{tsv_path}',
            delim='\t', header=true, quote='',
            columns={{ {cols_spec}, 'count': 'BIGINT' }}
        )
    """)

    nrows = db.execute("SELECT COUNT(*) FROM fivegrams").fetchone()[0]
    db_size = os.path.getsize(db_path) / 1e6
    db.close()
    print(f"  {nrows:,} rows, {db_size:.1f} MB, took {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract n-gram counts from corpus using KenLM")
    parser.add_argument("corpus", help="Input text file (one sentence per line)")
    parser.add_argument("--order", type=int, default=5,
                        help="N-gram order (default: 5)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Only keep n-grams with count >= this (default: 1)")
    parser.add_argument("--kenlm-bin", default=None,
                        help="Path to KenLM bin directory containing count_ngrams")
    parser.add_argument("--memory", default="80%%",
                        help="Sorting memory for KenLM (default: 80%%)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory (default: current dir)")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="Skip DuckDB output")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep intermediate binary files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tsv_path = os.path.join(args.out_dir, f"{args.order}gram_counts.tsv")
    db_path = os.path.join(args.out_dir, f"{args.order}grams.duckdb")

    # Use a temp dir for KenLM intermediate files
    with tempfile.TemporaryDirectory(prefix="kenlm_") as tmp_dir:
        if args.keep_tmp:
            tmp_dir = os.path.join(args.out_dir, "kenlm_tmp")
            os.makedirs(tmp_dir, exist_ok=True)

        # Step 1: run count_ngrams
        counts_path, vocab_path = run_count_ngrams(
            args.corpus, args.order, args.kenlm_bin, args.memory, tmp_dir
        )

        # Step 2: load vocab
        vocab = load_vocab(vocab_path)

        # Step 3: decode binary → TSV
        written = decode_binary(
            counts_path, vocab, args.order, args.min_count, tsv_path
        )

    # Step 4: load into DuckDB
    if not args.no_duckdb and written > 0:
        load_duckdb(tsv_path, db_path, args.order)

    print(f"\nDone!")
    print(f"  TSV:    {tsv_path}")
    if not args.no_duckdb:
        print(f"  DuckDB: {db_path}")


if __name__ == "__main__":
    main()
