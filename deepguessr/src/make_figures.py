"""Generate report figures from results CSVs/JSONs. Saves PNG (300 dpi) + PDF.

Figures:
  fig_error_cdf        - fraction within X km (log x) for every model
  fig_per_continent    - per-continent median km bar chart
  fig_training_curves  - train loss + val median km for regression heads
  fig_prediction_map   - GT vs pred for ~300 test points, one panel per model
  fig_data_distribution- train image locations (subsampled)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config
from geo import haversine_km

THRESHOLDS = [25, 200, 750, 2500]

# run_name -> display label, for models we plot.
MODELS = [
    ("zeroshot_single", "Zero-shot (single)"),
    ("zeroshot_ensemble", "Zero-shot (ensemble)"),
    ("regression_linear", "Regression (linear)"),
    ("regression_mlp", "Regression (MLP)"),
    ("geocell_mlp_onehot", "Geocell MLP (one-hot)"),
    ("geocell_mlp_smooth_tau75", "Geocell MLP (smooth)"),
    ("lora_geocell_smooth_tau75", "LoRA geocell (smooth)"),
    ("sanity_train_mean", "Train-mean"),
]


def save(fig, figures_dir: Path, name: str):
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def load_preds(results_dir: Path, run: str) -> pd.DataFrame | None:
    p = results_dir / f"{run}_preds.csv"
    return pd.read_csv(p) if p.exists() else None


def fig_error_cdf(results_dir, figures_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.logspace(0, 4.3, 200)  # 1 km .. ~20000 km
    for run, label in MODELS:
        df = load_preds(results_dir, run)
        if df is None:
            continue
        err = haversine_km(df["gt_lat"].to_numpy(), df["gt_lon"].to_numpy(),
                           df["pred_lat"].to_numpy(), df["pred_lon"].to_numpy())
        cdf = [(err <= x).mean() for x in xs]
        ax.plot(xs, cdf, label=label, lw=2)
    for t in THRESHOLDS:
        ax.axvline(t, color="gray", ls=":", lw=1)
        ax.text(t, 1.01, f"{t}", rotation=90, fontsize=8, va="bottom", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Distance threshold (km, log scale)")
    ax.set_ylabel("Fraction of test images within threshold")
    ax.set_title("Error CDF")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, figures_dir, "fig_error_cdf")


def fig_per_continent(results_dir, figures_dir):
    data = {}
    for run, label in MODELS:
        p = results_dir / f"{run}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        if "per_continent" in d:
            data[label] = d["per_continent"]
    if not data:
        return
    conts = sorted({c for pc in data.values() for c in pc})
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(data)
    x = np.arange(len(conts))
    for i, (label, pc) in enumerate(data.items()):
        vals = [pc[c]["median_km"] if c in pc else 0 for c in conts]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.set_xticklabels(conts, rotation=20, ha="right")
    ax.set_ylabel("Median error (km)")
    ax.set_title("Per-continent median error (by ground-truth continent)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, figures_dir, "fig_per_continent")


def fig_training_curves(results_dir, figures_dir):
    curves = [("regression_linear_curve.csv", "Regression (linear)", "train_loss_km"),
              ("regression_mlp_curve.csv", "Regression (MLP)", "train_loss_km"),
              ("geocell_mlp_onehot_curve.csv", "Geocell MLP (one-hot)", "train_loss"),
              ("geocell_mlp_smooth_tau75_curve.csv", "Geocell MLP (smooth)", "train_loss"),
              ("lora_geocell_smooth_tau75_curve.csv", "LoRA geocell (smooth)", "train_loss")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for fname, label, loss_col in curves:
        p = results_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if loss_col in df:
            ax1.plot(df["epoch"], df[loss_col], label=label, lw=2)
        ax2.plot(df["epoch"], df["val_median_km"], label=label, lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss")
    ax1.set_title("Training loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val median error (km)")
    ax2.set_title("Validation median error"); ax2.legend(); ax2.grid(alpha=0.3)
    save(fig, figures_dir, "fig_training_curves")


def fig_prediction_map(results_dir, figures_dir, n=300, seed=42):
    runs = [("zeroshot_ensemble", "Zero-shot (ensemble)"),
            ("regression_mlp", "Regression (MLP)"),
            ("geocell_mlp_smooth_tau75", "Geocell MLP (smooth)"),
            ("lora_geocell_smooth_tau75", "LoRA geocell (smooth)")]
    runs = [r for r in runs if (results_dir / f"{r[0]}_preds.csv").exists()]
    fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 6))
    if len(runs) == 1:
        axes = [axes]
    rng = np.random.default_rng(seed)
    for ax, (run, label) in zip(axes, runs):
        df = load_preds(results_dir, run)
        if df is None:
            continue
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        sub = df.iloc[idx]
        for _, r in sub.iterrows():
            ax.plot([r["gt_lon"], r["pred_lon"]], [r["gt_lat"], r["pred_lat"]],
                    color="gray", lw=0.3, alpha=0.5, zorder=1)
        ax.scatter(sub["gt_lon"], sub["gt_lat"], s=10, c="tab:blue",
                   label="GT", zorder=2)
        ax.scatter(sub["pred_lon"], sub["pred_lat"], s=12, c="tab:red",
                   marker="x", label="Pred", zorder=3)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(label); ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.suptitle("Ground truth vs prediction (300 random test images)")
    save(fig, figures_dir, "fig_prediction_map")


def fig_data_distribution(cfg, figures_dir, max_points=8000, seed=42):
    data_dir = Path(cfg["paths"]["data_dir"])
    p = data_dir / "train.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if len(df) > max_points:
        df = df.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(df["lon"], df["lat"], s=4, alpha=0.4, c="tab:green")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Train image locations (n={len(df)} subsample)")
    ax.grid(alpha=0.3)
    save(fig, figures_dir, "fig_data_distribution")


def fig_geocells(cfg, figures_dir):
    data_dir = Path(cfg["paths"]["data_dir"])
    tag = cfg.get("geocells", {}).get("tag", "default")
    p = data_dir / f"geocells_{tag}.json"
    if not p.exists():
        return
    gc = json.load(open(p))
    cells = gc["cells"]
    lons = [c["center_lon"] for c in cells]
    lats = [c["center_lat"] for c in cells]
    cnts = np.array([c["count"] for c in cells])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sizes = 20 + 180 * (cnts / cnts.max())
    sc = ax.scatter(lons, lats, s=sizes, c=cnts, cmap="viridis",
                    alpha=0.8, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="train images per cell")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Adaptive geocells (K={gc['K']}): center + size by count")
    ax.grid(alpha=0.3)
    save(fig, figures_dir, "fig_geocells")


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("Generating figures:")
    fig_error_cdf(results_dir, figures_dir)
    fig_per_continent(results_dir, figures_dir)
    fig_training_curves(results_dir, figures_dir)
    fig_prediction_map(results_dir, figures_dir)
    fig_data_distribution(cfg, figures_dir)
    fig_geocells(cfg, figures_dir)
    print(f"Figures -> {figures_dir}")


if __name__ == "__main__":
    main()
