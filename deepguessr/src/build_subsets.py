"""Build train/val/test subset CSVs from downloaded images + metadata.

- train.csv is ~2.9 GB: read in chunks, keep only rows whose id was unzipped.
- Seeded cap to target_train_images; val = seeded 5% of train.
- Test: seeded fixed subset of test_subset_size; ids saved to
  data/test_subset_ids.npy so every model uses the same test set.
- Writes data/{train,val,test}.csv with id,image_path,lat,lon,country_code
  and results/data_stats.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_dirs, load_config, set_seed
from geo import continents_of

CHUNK = 200_000


def downloaded_ids(images_dir: Path, split: str) -> set[int]:
    return {int(p.stem) for p in (images_dir / split).glob("*.jpg")}


def filter_csv_to_ids(csv_path: Path, ids: set[int], ds_cfg: dict) -> pd.DataFrame:
    """Stream the (possibly huge) CSV, keep rows whose id is in `ids`."""
    id_col = ds_cfg["id_col"]
    keep_cols = [id_col, ds_cfg["lat_col"], ds_cfg["lon_col"], ds_cfg["country_col"]]
    parts = []
    for chunk in pd.read_csv(csv_path, usecols=keep_cols, chunksize=CHUNK):
        sel = chunk[chunk[id_col].isin(ids)]
        if len(sel):
            parts.append(sel)
    if not parts:
        return pd.DataFrame(columns=keep_cols)
    return pd.concat(parts, ignore_index=True)


def normalize(df: pd.DataFrame, ds_cfg: dict, images_dir: Path, split: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df[ds_cfg["id_col"]].astype(np.int64)
    out["lat"] = df[ds_cfg["lat_col"]].astype(float)
    out["lon"] = df[ds_cfg["lon_col"]].astype(float)
    out["country_code"] = df[ds_cfg["country_col"]].astype(str).str.upper()
    out["image_path"] = out["id"].map(
        lambda i: str((images_dir / split / f"{i}.jpg").resolve())
    )
    out = out.dropna(subset=["lat", "lon"])
    return out.reset_index(drop=True)


def stats_block(df: pd.DataFrame) -> dict:
    countries = df["country_code"].tolist()
    conts = continents_of(countries)
    top20 = Counter(countries).most_common(20)
    cont_counts = Counter(conts)
    return {
        "n": int(len(df)),
        "n_countries": int(df["country_code"].nunique()),
        "top20_countries": top20,
        "per_continent": dict(cont_counts),
    }


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    ds = cfg["dataset"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    images_dir = Path(cfg["paths"]["images_dir"])
    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])

    rng = np.random.default_rng(cfg["seed"])

    # ---- TRAIN (+VAL) ----
    train_ids = downloaded_ids(images_dir, "train")
    print(f"Downloaded train images: {len(train_ids)}")
    train_meta = filter_csv_to_ids(raw_dir / "train.csv", train_ids, ds)
    train_meta = normalize(train_meta, ds, images_dir, "train")
    print(f"Matched train rows: {len(train_meta)}")

    # Seeded cap to target.
    target = cfg["download"]["target_train_images"]
    if len(train_meta) > target:
        idx = rng.choice(len(train_meta), size=target, replace=False)
        train_meta = train_meta.iloc[np.sort(idx)].reset_index(drop=True)

    # Val = 5% seeded.
    val_frac = cfg["subsets"]["val_frac"]
    n_val = int(round(len(train_meta) * val_frac))
    perm = rng.permutation(len(train_meta))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    val_df = train_meta.iloc[val_idx].reset_index(drop=True)
    train_df = train_meta.iloc[train_idx].reset_index(drop=True)

    # ---- TEST ----
    test_ids = downloaded_ids(images_dir, "test")
    print(f"Downloaded test images: {len(test_ids)}")
    test_meta = filter_csv_to_ids(raw_dir / "test.csv", test_ids, ds)
    test_meta = normalize(test_meta, ds, images_dir, "test")
    print(f"Matched test rows: {len(test_meta)}")

    test_size = min(cfg["download"]["test_subset_size"], len(test_meta))
    tidx = np.sort(rng.choice(len(test_meta), size=test_size, replace=False))
    test_df = test_meta.iloc[tidx].reset_index(drop=True)
    np.save(data_dir / "test_subset_ids.npy", test_df["id"].to_numpy())

    # ---- WRITE ----
    cols = ["id", "image_path", "lat", "lon", "country_code"]
    train_df[cols].to_csv(data_dir / "train.csv", index=False)
    val_df[cols].to_csv(data_dir / "val.csv", index=False)
    test_df[cols].to_csv(data_dir / "test.csv", index=False)

    stats = {
        "train": stats_block(train_df),
        "val": stats_block(val_df),
        "test": stats_block(test_df),
        "seed": cfg["seed"],
    }
    with open(results_dir / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== SUBSET SIZES ===")
    for split in ("train", "val", "test"):
        s = stats[split]
        print(f"{split}: n={s['n']}, countries={s['n_countries']}")
    print(f"\nStats -> {results_dir / 'data_stats.json'}")


if __name__ == "__main__":
    main()
