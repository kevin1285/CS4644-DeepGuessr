"""Inspect the OSV-5M dataset on Hugging Face.

Step 1 of the handoff: list repo files and confirm the metadata schema
(id / lat / lon / country columns) BEFORE writing any download logic.
Do not assume column names.
"""
from __future__ import annotations

import io
import sys

import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID = "osv5m/osv5m"
REPO_TYPE = "dataset"


def main() -> None:
    print(f"Listing files in {REPO_ID} ({REPO_TYPE}) ...")
    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"Total files: {len(files)}\n")

    # Group by top-level prefix for a readable overview.
    csvs = [f for f in files if f.endswith(".csv")]
    zips = [f for f in files if f.endswith(".zip")]
    others = [f for f in files if not f.endswith((".csv", ".zip"))]

    print("=== CSV files ===")
    for f in csvs:
        print(" ", f)

    print(f"\n=== ZIP shards ({len(zips)}) ===")
    # Show a sample of shard paths (train/test) rather than all of them.
    train_zips = [f for f in zips if "train" in f]
    test_zips = [f for f in zips if "test" in f]
    print(f"  train shards: {len(train_zips)} (sample: {train_zips[:5]})")
    print(f"  test shards:  {len(test_zips)} (sample: {test_zips[:5]})")

    print("\n=== Other files ===")
    for f in others[:40]:
        print(" ", f)

    # Inspect metadata schema from the root CSVs we care about.
    for csv_name in ("train.csv", "test.csv"):
        if csv_name not in csvs:
            print(f"\n!! {csv_name} not found at repo root; available CSVs above.")
            continue
        print(f"\n=== Schema of {csv_name} ===")
        path = hf_hub_download(
            REPO_ID, filename=csv_name, repo_type=REPO_TYPE
        )
        # Read only the first chunk to get columns + a few rows cheaply.
        head = pd.read_csv(path, nrows=5)
        print("columns:", list(head.columns))
        print("dtypes:")
        print(head.dtypes)
        print("sample rows:")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(head)


if __name__ == "__main__":
    sys.exit(main())
