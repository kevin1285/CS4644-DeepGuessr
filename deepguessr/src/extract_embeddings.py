"""Extract frozen CLIP ViT-B/16 image embeddings for each split.

Saves data/emb_{split}.npy (float16, L2-normalized) and
data/emb_{split}_ids.npy in the same row order as data/{split}.csv.
Resumable: processes in chunks and skips chunks already saved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip

from common import get_device, load_config, set_seed

CHUNK = 5000  # rows per resumable chunk


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.preprocess(img), self.ids[i], True
        except Exception:
            # Unreadable/corrupt: return zeros, flag invalid.
            return torch.zeros(3, 224, 224), self.ids[i], False


def extract_split(split, cfg, model, preprocess, device):
    data_dir = Path(cfg["paths"]["data_dir"])
    csv = data_dir / f"{split}.csv"
    if not csv.exists():
        print(f"[{split}] no {csv}, skipping")
        return
    df = pd.read_csv(csv)
    emb_path = data_dir / f"emb_{split}.npy"
    ids_path = data_dir / f"emb_{split}_ids.npy"

    if emb_path.exists() and ids_path.exists():
        existing_ids = np.load(ids_path)
        if len(existing_ids) == len(df) and np.array_equal(
            existing_ids, df["id"].to_numpy()
        ):
            print(f"[{split}] embeddings already complete ({len(df)}), skipping")
            return

    bs = cfg["clip"]["batch_size"]
    nw = cfg["clip"]["num_workers"]
    all_emb = np.zeros((len(df), model.visual.output_dim), dtype=np.float16)
    all_ids = df["id"].to_numpy()

    for start in range(0, len(df), CHUNK):
        end = min(start + CHUNK, len(df))
        sub = df.iloc[start:end].reset_index(drop=True)
        ds = ImageDataset(sub, preprocess)
        loader = DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=True)
        offset = start
        pbar = tqdm(loader, desc=f"[{split}] {start}:{end}")
        with torch.no_grad():
            for imgs, ids, valid in pbar:
                imgs = imgs.to(device)
                if device == "mps":
                    feats = model.encode_image(imgs)
                else:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats = feats.float().cpu().numpy().astype(np.float16)
                n = feats.shape[0]
                all_emb[offset : offset + n] = feats
                offset += n

    assert not np.isnan(all_emb.astype(np.float32)).any(), "NaNs in embeddings!"
    np.save(emb_path, all_emb)
    np.save(ids_path, all_ids)
    print(f"[{split}] saved {all_emb.shape} -> {emb_path}")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    print(f"Device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"]
    )
    model = model.to(device).eval()
    print(f"Loaded CLIP {cfg['clip']['model']}/{cfg['clip']['pretrained']}, "
          f"emb dim {model.visual.output_dim}")

    for split in ("train", "val", "test"):
        extract_split(split, cfg, model, preprocess, device)


if __name__ == "__main__":
    main()
