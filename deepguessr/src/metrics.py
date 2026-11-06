"""Evaluation metrics shared by every model.

evaluate(...) -> dict with median/mean km, acc@{25,200,750,2500} km,
country_acc, and n. per_continent(...) groups by GT continent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from geo import continents_of, country_of, haversine_km

DEFAULT_THRESHOLDS = (25, 200, 750, 2500)


def evaluate(
    pred_lat,
    pred_lon,
    gt_lat,
    gt_lon,
    pred_country=None,
    gt_country=None,
    thresholds_km=DEFAULT_THRESHOLDS,
) -> dict:
    pred_lat = np.asarray(pred_lat, dtype=np.float64)
    pred_lon = np.asarray(pred_lon, dtype=np.float64)
    gt_lat = np.asarray(gt_lat, dtype=np.float64)
    gt_lon = np.asarray(gt_lon, dtype=np.float64)

    err = haversine_km(gt_lat, gt_lon, pred_lat, pred_lon)

    out: dict = {
        "n": int(err.size),
        "median_km": float(np.median(err)),
        "mean_km": float(np.mean(err)),
    }
    for t in thresholds_km:
        out[f"acc@{t}km"] = float(np.mean(err <= t))

    # Country accuracy: derive via reverse geocoding when not supplied.
    if gt_country is None:
        gt_country = country_of(gt_lat, gt_lon)
    if pred_country is None:
        pred_country = country_of(pred_lat, pred_lon)
    gt_country = np.asarray(gt_country)
    pred_country = np.asarray(pred_country)
    out["country_acc"] = float(np.mean(gt_country == pred_country))
    return out


def per_continent(
    pred_lat,
    pred_lon,
    gt_lat,
    gt_lon,
    pred_country=None,
    gt_country=None,
) -> dict:
    pred_lat = np.asarray(pred_lat, dtype=np.float64)
    pred_lon = np.asarray(pred_lon, dtype=np.float64)
    gt_lat = np.asarray(gt_lat, dtype=np.float64)
    gt_lon = np.asarray(gt_lon, dtype=np.float64)

    err = haversine_km(gt_lat, gt_lon, pred_lat, pred_lon)
    if gt_country is None:
        gt_country = country_of(gt_lat, gt_lon)
    if pred_country is None:
        pred_country = country_of(pred_lat, pred_lon)
    gt_country = np.asarray(gt_country)
    pred_country = np.asarray(pred_country)
    gt_cont = np.asarray(continents_of(gt_country.tolist()))

    groups: dict[str, dict] = {}
    for cont in sorted(set(gt_cont.tolist())):
        mask = gt_cont == cont
        groups[cont] = {
            "n": int(mask.sum()),
            "median_km": float(np.median(err[mask])),
            "country_acc": float(np.mean(gt_country[mask] == pred_country[mask])),
        }
    return groups
