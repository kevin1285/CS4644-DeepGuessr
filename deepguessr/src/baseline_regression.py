"""Baseline B - Regression on frozen CLIP embeddings.

Model predicts a 3D vector -> normalize to unit sphere -> lat/lon (avoids the
+/-180 wraparound). Loss = mean haversine km (differentiable). Trains a linear
head and an MLP head; reports the better on val, evaluates best on test.
Logs train loss + val median km per epoch to a CSV for the training-curve figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import get_device, load_config, save_run, set_seed
from geo import haversine_km
from metrics import evaluate, per_continent


def load_emb(data_dir: Path, split: str, csv: pd.DataFrame):
    emb = np.load(data_dir / f"emb_{split}.npy").astype(np.float32)
    ids = np.load(data_dir / f"emb_{split}_ids.npy")
    assert np.array_equal(ids, csv["id"].to_numpy()), f"{split} emb/id mismatch"
    return emb


def unit_to_latlon_torch(xyz):
    xyz = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(y, x))
    return lat, lon


class LinearHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )

    def forward(self, x):
        return self.net(x)


def train_head(head, tr_x, tr_y, va_x, va_latlon, cfg, device, tag):
    rcfg = cfg["regression"]
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=rcfg["lr"],
                            weight_decay=rcfg["weight_decay"])
    epochs = rcfg["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = rcfg["batch_size"]

    tr_x = torch.from_numpy(tr_x).to(device)
    tr_lat = torch.from_numpy(tr_y[0]).to(device)
    tr_lon = torch.from_numpy(tr_y[1]).to(device)
    va_x_t = torch.from_numpy(va_x).to(device)
    va_lat, va_lon = va_latlon

    n = tr_x.shape[0]
    best_med = float("inf")
    best_state = None
    log = []

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xyz = head(tr_x[idx])
            plat, plon = unit_to_latlon_torch(xyz)
            loss = haversine_km(tr_lat[idx], tr_lon[idx], plat, plon).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        train_loss = total / n

        head.eval()
        with torch.no_grad():
            xyz = head(va_x_t)
            plat, plon = unit_to_latlon_torch(xyz)
            plat = plat.cpu().numpy()
            plon = plon.cpu().numpy()
        va_med = float(np.median(haversine_km(va_lat, va_lon, plat, plon)))
        log.append({"epoch": ep, "train_loss_km": train_loss, "val_median_km": va_med})
        if va_med < best_med:
            best_med = va_med
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        print(f"[{tag}] ep {ep:02d} train_km={train_loss:7.1f} val_med_km={va_med:7.1f}")

    head.load_state_dict(best_state)
    return head, best_med, pd.DataFrame(log)


def predict(head, x, device):
    head.eval()
    with torch.no_grad():
        xyz = head(torch.from_numpy(x).to(device))
        plat, plon = unit_to_latlon_torch(xyz)
    return plat.cpu().numpy(), plon.cpu().numpy()


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    tr_x = load_emb(data_dir, "train", train)
    va_x = load_emb(data_dir, "val", val)
    te_x = load_emb(data_dir, "test", test)
    dim = tr_x.shape[1]

    tr_y = (train["lat"].to_numpy().astype(np.float32),
            train["lon"].to_numpy().astype(np.float32))
    va_latlon = (val["lat"].to_numpy(), val["lon"].to_numpy())

    heads = {
        "linear": LinearHead(dim),
        "mlp": MLPHead(dim, cfg["regression"]["hidden"]),
    }

    results = {}
    for name, head in heads.items():
        set_seed(cfg["seed"])
        trained, best_med, log = train_head(
            head, tr_x, tr_y, va_x, va_latlon, cfg, device, f"reg_{name}")
        log.to_csv(results_dir / f"regression_{name}_curve.csv", index=False)
        results[name] = (trained, best_med)

    # Pick better head on val median km.
    best_name = min(results, key=lambda k: results[k][1])
    print(f"\nBest regression head on val: {best_name} "
          f"(val median {results[best_name][1]:.1f} km)")

    gt_lat = test["lat"].to_numpy()
    gt_lon = test["lon"].to_numpy()
    gt_country = test["country_code"].to_numpy()

    for name, (trained, _) in results.items():
        plat, plon = predict(trained, te_x, device)
        m = evaluate(plat, plon, gt_lat, gt_lon, gt_country=gt_country)
        print(f"[reg_{name}] TEST median_km={m['median_km']:.0f} "
              f"acc@200={m['acc@200km']:.3f} country_acc={m['country_acc']:.3f}")
        preds = pd.DataFrame({
            "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
            "pred_lat": plat, "pred_lon": plon, "gt_country": gt_country,
        })
        save_run(cfg, f"regression_{name}", m, preds, extra={
            "per_continent": per_continent(plat, plon, gt_lat, gt_lon,
                                            gt_country=gt_country),
            "selected_on_val": (name == best_name),
            "loss": cfg["regression"]["loss"],
        })


if __name__ == "__main__":
    main()
