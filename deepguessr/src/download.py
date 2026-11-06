"""Download a small subset of OSV-5M: selected zip shards + metadata CSVs,
then unzip images into data/images/{train,test}/{id}.jpg.

Resumable: skips shards already unzipped. Scales via configs/default.yaml.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from common import ensure_dirs, load_config


def download_shard(repo_id: str, repo_type: str, split: str, shard: int, raw_dir: Path) -> Path:
    fname = f"images/{split}/{shard:02d}.zip"
    local = hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        repo_type=repo_type,
        local_dir=str(raw_dir),
    )
    return Path(local)


def unzip_shard(zip_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".jpg")]
        for m in members:
            # Flatten to {out_dir}/{basename}.jpg
            name = Path(m).name
            target = out_dir / name
            if target.exists():
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            count += 1
    return count


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    ds = cfg["dataset"]
    dl = cfg["download"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    images_dir = Path(cfg["paths"]["images_dir"])

    # Metadata CSVs at repo root.
    for csv_name in ("train.csv", "test.csv"):
        print(f"Downloading {csv_name} ...")
        hf_hub_download(
            repo_id=ds["repo_id"],
            filename=csv_name,
            repo_type=ds["repo_type"],
            local_dir=str(raw_dir),
        )

    for split, shards in (("train", dl["train_shards"]), ("test", dl["test_shards"])):
        out_dir = images_dir / split
        for shard in shards:
            print(f"\n[{split}] shard {shard:02d}: downloading ...")
            zip_path = download_shard(
                ds["repo_id"], ds["repo_type"], split, shard, raw_dir
            )
            print(f"[{split}] shard {shard:02d}: unzipping ...")
            n = unzip_shard(zip_path, out_dir)
            print(f"[{split}] shard {shard:02d}: wrote {n} new images")

    n_train = len(list((images_dir / "train").glob("*.jpg")))
    n_test = len(list((images_dir / "test").glob("*.jpg")))
    print(f"\nDone. images/train: {n_train} jpg, images/test: {n_test} jpg")


if __name__ == "__main__":
    main()
