"""Trivial sanity baselines to confirm the metric code is sane.

- train-mean: always predict the mean train location (mean of unit vectors).
- most-common-country: always predict the center of the most frequent country.
Median error should be thousands of km.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, save_run
from geo import mean_latlon
from metrics import evaluate, per_continent


def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    gt_lat = test["lat"].to_numpy()
    gt_lon = test["lon"].to_numpy()
    gt_country = test["country_code"].to_numpy()
    n = len(test)

    # --- train-mean ---
    mlat, mlon = mean_latlon(train["lat"].to_numpy(), train["lon"].to_numpy())
    pred_lat = np.full(n, mlat)
    pred_lon = np.full(n, mlon)
    m = evaluate(pred_lat, pred_lon, gt_lat, gt_lon, gt_country=gt_country)
    print(f"[train-mean] center=({mlat:.3f},{mlon:.3f}) "
          f"median_km={m['median_km']:.0f} country_acc={m['country_acc']:.3f}")
    preds = pd.DataFrame({
        "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
        "pred_lat": pred_lat, "pred_lon": pred_lon,
    })
    save_run(cfg, "sanity_train_mean", m, preds,
             extra={"per_continent": per_continent(
                 pred_lat, pred_lon, gt_lat, gt_lon, gt_country=gt_country)})

    # --- most-common-country center ---
    top_country = Counter(train["country_code"]).most_common(1)[0][0]
    sub = train[train["country_code"] == top_country]
    clat, clon = mean_latlon(sub["lat"].to_numpy(), sub["lon"].to_numpy())
    pred_lat = np.full(n, clat)
    pred_lon = np.full(n, clon)
    pred_country = np.full(n, top_country)
    m2 = evaluate(pred_lat, pred_lon, gt_lat, gt_lon,
                  pred_country=pred_country, gt_country=gt_country)
    print(f"[most-common-country] {top_country} center=({clat:.3f},{clon:.3f}) "
          f"median_km={m2['median_km']:.0f} country_acc={m2['country_acc']:.3f}")
    preds2 = pd.DataFrame({
        "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
        "pred_lat": pred_lat, "pred_lon": pred_lon,
        "gt_country": gt_country, "pred_country": pred_country,
    })
    save_run(cfg, "sanity_common_country", m2, preds2)


if __name__ == "__main__":
    main()
