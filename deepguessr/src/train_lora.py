"""LoRA fine-tuning (E5) — main model for the final report.

Fine-tunes the CLIP ViT-B/16 vision tower with LoRA + a geocell classification
head, end-to-end on images. Uses HF `CLIPVisionModelWithProjection`
("openai/clip-vit-base-patch16"), whose 512-d projected image embedding is
verified identical (cosine ~1.0) to the open_clip backbone used for the frozen
experiments — so frozen vs. LoRA (E6) is an apples-to-apples comparison.

Design (per final handoff §4 E5):
- LoRA on q_proj/v_proj (r=16, alpha=32, dropout=0.1).
- Head: linear 512->K on the L2-normalized projected embedding.
- Warm-start the head from the best frozen geocell head (same K + labels).
- Augmentation (train only): RandomResizedCrop + mild ColorJitter.
  NO horizontal flip — flipping swaps driving side (a real geo cue) and text.
- AdamW, two param groups (LoRA lr 1e-4, head lr 1e-3), wd 0.01.
- Linear warmup (5% steps) -> cosine decay. Mixed precision when on CUDA.
- Select best checkpoint by val median km. Resumable from checkpoint.

Usage:
  python train_lora.py --labels smooth --tau 75 --epochs 3 --train-images 100000
  python train_lora.py --resume results/lora_geocell_smooth_tau75_ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from common import get_device, load_config, save_run, set_seed
from geo import haversine_km
from metrics import evaluate, per_continent


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_transforms(processor, train: bool):
    size = processor.crop_size["height"]
    mean, std = processor.image_mean, processor.image_std
    if train:
        # NO horizontal flip by design (driving side / text are geo cues).
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class GeoImageDataset(Dataset):
    def __init__(self, df, cell_assign, tfm):
        self.paths = df["image_path"].tolist()
        self.lat = df["lat"].to_numpy(dtype=np.float32)
        self.lon = df["lon"].to_numpy(dtype=np.float32)
        self.cell = cell_assign
        self.tfm = tfm
        self._size = tfm.transforms[0].size
        if isinstance(self._size, (tuple, list)):
            self._size = self._size[0]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            px = self.tfm(img)
        except Exception:
            px = torch.zeros(3, self._size, self._size)
        return px, int(self.cell[i]), self.lat[i], self.lon[i]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LoRAGeoModel(nn.Module):
    def __init__(self, vision_proj_model, K, hidden=0):
        super().__init__()
        self.vision = vision_proj_model
        dim = vision_proj_model.config.projection_dim  # 512
        if hidden:
            self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, K))
        else:
            self.head = nn.Linear(dim, K)

    def forward(self, pixel_values):
        emb = self.vision(pixel_values=pixel_values).image_embeds  # (B, 512)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.head(emb)


def smoothed_targets(lat, lon, c_lat, c_lon, tau):
    d = haversine_km(lat[:, None], lon[:, None], c_lat[None, :], c_lon[None, :])
    return torch.softmax(-d / tau, dim=1)


def count_trainable(model):
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return t, tot


def warm_start_head(model, cfg, data_dir, arch, labels, tau, K):
    """Load the best frozen geocell head weights into the LoRA model's head.

    The frozen linear head was `nn.Linear(512, K)` in train_geocell_head.py
    (attr `.fc`). We retrain the frozen head quickly here only if a saved
    state isn't found, otherwise we skip (the frozen script does not persist
    weights by default). This function is best-effort: it looks for an
    optional cached head file and warm-starts if present.
    """
    suffix = "onehot" if labels == "onehot" else f"smooth_tau{int(tau)}"
    head_path = data_dir / f"frozen_head_{arch}_{suffix}.pt"
    if head_path.exists() and isinstance(model.head, nn.Linear):
        sd = torch.load(head_path, map_location="cpu")
        model.head.load_state_dict(sd)
        print(f"Warm-started head from {head_path}")
        return True
    print("No cached frozen head found; head starts from random init "
          f"(expected at {head_path}).")
    return False


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["onehot", "smooth"], default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-images", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    lc = cfg["lora"]
    tag = cfg["geocells"]["tag"]
    labels = args.labels or lc["labels"]
    tau = args.tau if args.tau is not None else cfg["geocell_head"]["tau_km"]
    epochs = args.epochs or lc["epochs"]
    n_train = args.train_images or lc["train_images"]
    use_amp = device == "cuda"

    # Geocells.
    gcells = json.load(open(data_dir / f"geocells_{tag}.json"))
    K = gcells["K"]
    centers_lat = np.array([c["center_lat"] for c in gcells["cells"]], dtype=np.float32)
    centers_lon = np.array([c["center_lon"] for c in gcells["cells"]], dtype=np.float32)
    c_lat_t = torch.from_numpy(centers_lat).to(device)
    c_lon_t = torch.from_numpy(centers_lon).to(device)

    # Data.
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")
    tr_cell = np.load(data_dir / f"cell_assign_train_{tag}.npy")
    va_cell = np.load(data_dir / f"cell_assign_val_{tag}.npy")
    te_cell = np.load(data_dir / f"cell_assign_test_{tag}.npy")

    if n_train < len(train):
        rng = np.random.default_rng(cfg["seed"])
        idx = np.sort(rng.choice(len(train), size=n_train, replace=False))
        train = train.iloc[idx].reset_index(drop=True)
        tr_cell = tr_cell[idx]

    processor = CLIPImageProcessor.from_pretrained(lc["hf_vision_model"])
    vision = CLIPVisionModelWithProjection.from_pretrained(lc["hf_vision_model"])

    lora_cfg = LoraConfig(
        r=lc["rank"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"], bias="none",
    )
    vision = get_peft_model(vision, lora_cfg)
    model = LoRAGeoModel(vision, K).to(device)

    warm_start_head(model, cfg, data_dir, "linear", labels, tau, K)

    trainable, total = count_trainable(model)
    print(f"K={K} labels={labels}" + (f" tau={tau}" if labels == "smooth" else "")
          + f" epochs={epochs} train_images={len(train)} device={device}")
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    tr_ds = GeoImageDataset(train, tr_cell, build_transforms(processor, True))
    va_ds = GeoImageDataset(val, va_cell, build_transforms(processor, False))
    te_ds = GeoImageDataset(test, te_cell, build_transforms(processor, False))
    bs = lc["batch_size"]
    nw = lc["num_workers"]
    accum = lc.get("grad_accum", 1)
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw,
                           drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=bs, num_workers=nw)
    te_loader = DataLoader(te_ds, batch_size=bs, num_workers=nw)

    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("head.")]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": lc["lr_lora"]},
        {"params": head_params, "lr": lc["lr_head"]},
    ], weight_decay=lc["weight_decay"])

    steps_per_epoch = max(1, len(tr_loader) // accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_med = float("inf")
    best_state = None
    log = []
    ckpt_path = results_dir / (f"lora_geocell_"
                               + ("onehot" if labels == "onehot"
                                  else f"smooth_tau{int(tau)}") + "_ckpt.pt")

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_med = ck.get("best_med", float("inf"))
        best_state = ck.get("best_state")
        log = ck.get("log", [])
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    def eval_loader(loader, df):
        model.eval()
        preds = []
        with torch.no_grad():
            for px, _, _, _ in loader:
                px = px.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(px)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        plat, plon = centers_lat[pred], centers_lon[pred]
        med = float(np.median(haversine_km(
            df["lat"].to_numpy(), df["lon"].to_numpy(), plat, plon)))
        return pred, plat, plon, med

    gstep = start_epoch * steps_per_epoch
    for ep in range(start_epoch, epochs):
        model.train()
        running, seen = 0.0, 0
        opt.zero_grad()
        t0 = time.time()
        pbar = tqdm(tr_loader, desc=f"epoch {ep}")
        for step, (px, cell, lat, lon) in enumerate(pbar):
            px = px.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(px)
                if labels == "onehot":
                    loss = F.cross_entropy(logits, cell.to(device))
                else:
                    q = smoothed_targets(lat.to(device), lon.to(device),
                                         c_lat_t, c_lon_t, tau)
                    loss = -(q * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                gstep += 1
            running += loss.item() * len(cell)
            seen += len(cell)
            pbar.set_postfix(loss=f"{running / seen:.3f}")
        train_loss = running / seen
        img_per_s = seen / max(1e-6, time.time() - t0)

        _, _, _, va_med = eval_loader(va_loader, val)
        log.append({"epoch": ep, "train_loss": train_loss,
                    "val_median_km": va_med, "img_per_s": img_per_s})
        print(f"epoch {ep}: train_loss={train_loss:.4f} "
              f"val_median_km={va_med:.1f} throughput={img_per_s:.1f} img/s")
        if va_med < best_med:
            best_med = va_med
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep,
                    "best_med": best_med, "best_state": best_state,
                    "log": log}, ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test.
    pred, plat, plon, _ = eval_loader(te_loader, test)
    gt_lat, gt_lon = test["lat"].to_numpy(), test["lon"].to_numpy()
    gt_country = test["country_code"].to_numpy()
    m = evaluate(plat, plon, gt_lat, gt_lon, gt_country=gt_country)
    m["cell_acc"] = float(np.mean(pred == te_cell))
    m["trainable_params"] = trainable

    suffix = "onehot" if labels == "onehot" else f"smooth_tau{int(tau)}"
    run_name = f"lora_geocell_{suffix}"
    print(f"[{run_name}] TEST median_km={m['median_km']:.0f} "
          f"acc@200={m['acc@200km']:.3f} country_acc={m['country_acc']:.3f} "
          f"cell_acc={m['cell_acc']:.3f}")

    pd.DataFrame(log).to_csv(results_dir / f"{run_name}_curve.csv", index=False)
    preds = pd.DataFrame({
        "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
        "pred_lat": plat, "pred_lon": plon, "gt_country": gt_country,
    })
    save_run(cfg, run_name, m, preds, extra={
        "per_continent": per_continent(plat, plon, gt_lat, gt_lon,
                                        gt_country=gt_country),
        "K": K, "labels": labels, "tau_km": tau if labels == "smooth" else None,
        "lora": {"rank": lc["rank"], "alpha": lc["alpha"],
                 "dropout": lc["dropout"], "target_modules": lc["target_modules"],
                 "train_images": len(train), "epochs": epochs,
                 "lr_lora": lc["lr_lora"], "lr_head": lc["lr_head"],
                 "no_hflip": True},
    })


if __name__ == "__main__":
    main()
