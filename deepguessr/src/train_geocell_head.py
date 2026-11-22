"""Main method (early) - geocell classifier on frozen CLIP embeddings.

Head: linear 512->K (or MLP 512->hidden->K). Two label modes:
  --labels onehot : standard cross-entropy with the true cell.
  --labels smooth : target q_k ∝ exp(-haversine(c_k, y)/tau), soft cross-entropy.
Prediction = center of argmax cell; country = reverse-geocode predicted point.
Reports median km, acc@{25,200,750,2500}, country acc, cell acc, top-5 cell acc.

Usage:
  python train_geocell_head.py --labels onehot
  python train_geocell_head.py --labels smooth --tau 75
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import get_device, load_config, save_run, set_seed
from geo import haversine_km
from metrics import evaluate, per_continent


def load_emb(data_dir, split, csv):
    emb = np.load(data_dir / f"emb_{split}.npy").astype(np.float32)
    ids = np.load(data_dir / f"emb_{split}_ids.npy")
    assert np.array_equal(ids, csv["id"].to_numpy()), f"{split} emb/id mismatch"
    return emb


class LinearHead(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.fc = nn.Linear(dim, K)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, dim, hidden, K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K))

    def forward(self, x):
        return self.net(x)


def smoothed_targets(batch_lat, batch_lon, centers_lat, centers_lon, tau):
    """q_k ∝ exp(-d(c_k, y)/tau), normalized. Shapes: (B,) and (K,) -> (B,K)."""
    d = haversine_km(batch_lat[:, None], batch_lon[:, None],
                     centers_lat[None, :], centers_lon[None, :])  # (B,K)
    logits = -d / tau
    return torch.softmax(logits, dim=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", choices=["onehot", "smooth"], default="onehot")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--arch", choices=["linear", "mlp"], default=None)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    hc = cfg["geocell_head"]
    tag = cfg["geocells"]["tag"]
    arch = args.arch or hc["arch"]
    tau = args.tau if args.tau is not None else hc["tau_km"]

    # Load geocells + assignments.
    gcells = json.load(open(data_dir / f"geocells_{tag}.json"))
    K = gcells["K"]
    centers_lat = np.array([c["center_lat"] for c in gcells["cells"]])
    centers_lon = np.array([c["center_lon"] for c in gcells["cells"]])
    print(f"K={K} cells, arch={arch}, labels={args.labels}"
          + (f", tau={tau}" if args.labels == "smooth" else ""))

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    tr_x = torch.from_numpy(load_emb(data_dir, "train", train)).to(device)
    va_x = torch.from_numpy(load_emb(data_dir, "val", val)).to(device)
    te_x = torch.from_numpy(load_emb(data_dir, "test", test)).to(device)
    dim = tr_x.shape[1]

    tr_cell = torch.from_numpy(
        np.load(data_dir / f"cell_assign_train_{tag}.npy")).to(device)
    tr_lat = torch.from_numpy(train["lat"].to_numpy().astype(np.float32)).to(device)
    tr_lon = torch.from_numpy(train["lon"].to_numpy().astype(np.float32)).to(device)
    c_lat_t = torch.from_numpy(centers_lat.astype(np.float32)).to(device)
    c_lon_t = torch.from_numpy(centers_lon.astype(np.float32)).to(device)

    head = (LinearHead(dim, K) if arch == "linear"
            else MLPHead(dim, hc["hidden"], K)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hc["lr"],
                            weight_decay=hc["weight_decay"])
    epochs = hc["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = hc["batch_size"]
    n = tr_x.shape[0]

    va_lat = val["lat"].to_numpy()
    va_lon = val["lon"].to_numpy()

    best_med = float("inf")
    best_state = None
    log = []

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = head(tr_x[idx])
            if args.labels == "onehot":
                loss = F.cross_entropy(logits, tr_cell[idx])
            else:
                q = smoothed_targets(tr_lat[idx], tr_lon[idx], c_lat_t, c_lon_t, tau)
                logp = F.log_softmax(logits, dim=1)
                loss = -(q * logp).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        train_loss = total / n

        head.eval()
        with torch.no_grad():
            pred = head(va_x).argmax(dim=1).cpu().numpy()
        plat = centers_lat[pred]
        plon = centers_lon[pred]
        va_med = float(np.median(haversine_km(va_lat, va_lon, plat, plon)))
        log.append({"epoch": ep, "train_loss": train_loss, "val_median_km": va_med})
        if va_med < best_med:
            best_med = va_med
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
        print(f"ep {ep:02d} loss={train_loss:.4f} val_med_km={va_med:7.1f}")

    head.load_state_dict(best_state)

    # Test evaluation.
    head.eval()
    with torch.no_grad():
        logits = head(te_x)
        pred = logits.argmax(dim=1).cpu().numpy()
        top5 = logits.topk(5, dim=1).indices.cpu().numpy()
    plat = centers_lat[pred]
    plon = centers_lon[pred]
    gt_lat = test["lat"].to_numpy()
    gt_lon = test["lon"].to_numpy()
    gt_country = test["country_code"].to_numpy()
    te_cell = np.load(data_dir / f"cell_assign_test_{tag}.npy")

    m = evaluate(plat, plon, gt_lat, gt_lon, gt_country=gt_country)
    m["cell_acc"] = float(np.mean(pred == te_cell))
    m["top5_cell_acc"] = float(np.mean([t in row for t, row in zip(te_cell, top5)]))

    suffix = "onehot" if args.labels == "onehot" else f"smooth_tau{int(tau)}"
    run_name = f"geocell_{arch}_{suffix}"
    print(f"[{run_name}] TEST median_km={m['median_km']:.0f} "
          f"acc@200={m['acc@200km']:.3f} country_acc={m['country_acc']:.3f} "
          f"cell_acc={m['cell_acc']:.3f} top5={m['top5_cell_acc']:.3f}")

    pd.DataFrame(log).to_csv(results_dir / f"{run_name}_curve.csv", index=False)

    # Persist the trained head so LoRA fine-tuning can warm-start from it.
    head_path = Path(cfg["paths"]["data_dir"]) / f"frozen_head_{arch}_{suffix}.pt"
    torch.save(head.state_dict(), head_path)
    print(f"Saved frozen head -> {head_path}")

    preds = pd.DataFrame({
        "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
        "pred_lat": plat, "pred_lon": plon, "gt_country": gt_country,
    })
    save_run(cfg, run_name, m, preds, extra={
        "per_continent": per_continent(plat, plon, gt_lat, gt_lon,
                                        gt_country=gt_country),
        "K": K, "labels": args.labels, "tau_km": tau if args.labels == "smooth" else None,
        "arch": arch,
    })


if __name__ == "__main__":
    main()
