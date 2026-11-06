"""Adaptive geocell construction (quadtree over train coordinates).

- Start from a coarse lon/lat grid.
- Recursively split any cell with > max_per_cell train images into 4 quadrants,
  stopping at a minimum cell side (degrees) to avoid infinite splitting.
- Drop cells with < min_per_cell images and reassign their images to the nearest
  remaining cell center.
- Cell center = mean of train images in the cell (mean of unit vectors).

Saves data/geocells_<tag>.json (bounds, center lat/lon, count) and per-split
cell assignments. Test assignment = nearest cell center (diagnostics only).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config
from geo import haversine_km, mean_latlon

# Coarse starting grid (handoff suggests ~8x4 over lon/lat).
N_LON0 = 8
N_LAT0 = 4


class Cell:
    __slots__ = ("lon_min", "lon_max", "lat_min", "lat_max", "idx")

    def __init__(self, lon_min, lon_max, lat_min, lat_max, idx):
        self.lon_min, self.lon_max = lon_min, lon_max
        self.lat_min, self.lat_max = lat_min, lat_max
        self.idx = idx  # numpy array of row indices into the train arrays

    @property
    def side(self):
        return min(self.lon_max - self.lon_min, self.lat_max - self.lat_min)

    def quadrants(self, lons, lats):
        mlon = 0.5 * (self.lon_min + self.lon_max)
        mlat = 0.5 * (self.lat_min + self.lat_max)
        out = []
        for (lo0, lo1), (la0, la1) in (
            ((self.lon_min, mlon), (self.lat_min, mlat)),
            ((mlon, self.lon_max), (self.lat_min, mlat)),
            ((self.lon_min, mlon), (mlat, self.lat_max)),
            ((mlon, self.lon_max), (mlat, self.lat_max)),
        ):
            sel = self.idx[
                (lons[self.idx] >= lo0) & (lons[self.idx] < lo1)
                & (lats[self.idx] >= la0) & (lats[self.idx] < la1)
            ]
            out.append(Cell(lo0, lo1, la0, la1, sel))
        return out


def build_quadtree(lons, lats, max_per_cell, min_cell_side):
    cells = []
    lon_edges = np.linspace(-180, 180, N_LON0 + 1)
    lat_edges = np.linspace(-90, 90, N_LAT0 + 1)
    all_idx = np.arange(len(lons))
    stack = []
    for i in range(N_LON0):
        for j in range(N_LAT0):
            lo0, lo1 = lon_edges[i], lon_edges[i + 1]
            la0, la1 = lat_edges[j], lat_edges[j + 1]
            sel = all_idx[
                (lons >= lo0) & (lons < lo1) & (lats >= la0) & (lats < la1)
            ]
            stack.append(Cell(lo0, lo1, la0, la1, sel))

    while stack:
        c = stack.pop()
        if len(c.idx) == 0:
            continue
        if len(c.idx) > max_per_cell and c.side / 2 >= min_cell_side:
            stack.extend(c.quadrants(lons, lats))
        else:
            cells.append(c)
    return cells


def assign_nearest(lats, lons, centers_lat, centers_lon):
    """Vectorized nearest-center assignment via haversine (chunked)."""
    n = len(lats)
    out = np.empty(n, dtype=np.int64)
    clat = np.asarray(centers_lat)
    clon = np.asarray(centers_lon)
    CH = 2000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        la = lats[s:e][:, None]
        lo = lons[s:e][:, None]
        d = haversine_km(la, lo, clat[None, :], clon[None, :])
        out[s:e] = d.argmin(axis=1)
    return out


def build_cells(train_df, max_per_cell, min_per_cell, min_cell_side):
    lons = train_df["lon"].to_numpy()
    lats = train_df["lat"].to_numpy()
    cells = build_quadtree(lons, lats, max_per_cell, min_cell_side)

    # Compute centers + counts; separate sparse cells.
    centers, counts, kept = [], [], []
    for c in cells:
        if len(c.idx) == 0:
            continue
        clat, clon = mean_latlon(lats[c.idx], lons[c.idx])
        if len(c.idx) >= min_per_cell:
            kept.append(c)
            centers.append((clat, clon))
            counts.append(len(c.idx))

    centers_lat = np.array([c[0] for c in centers])
    centers_lon = np.array([c[1] for c in centers])

    # Assign ALL train points to nearest kept center (handles dropped sparse cells).
    assign = assign_nearest(lats, lons, centers_lat, centers_lon)

    # Recompute centers as mean of assigned points (more accurate after reassignment).
    K = len(kept)
    final_lat = np.zeros(K)
    final_lon = np.zeros(K)
    final_cnt = np.zeros(K, dtype=int)
    for k in range(K):
        m = assign == k
        final_cnt[k] = int(m.sum())
        if final_cnt[k] > 0:
            final_lat[k], final_lon[k] = mean_latlon(lats[m], lons[m])
        else:
            final_lat[k], final_lon[k] = centers_lat[k], centers_lon[k]

    cell_records = [
        {
            "idx": k,
            "center_lat": float(final_lat[k]),
            "center_lon": float(final_lon[k]),
            "count": int(final_cnt[k]),
            "bounds": [float(kept[k].lon_min), float(kept[k].lon_max),
                       float(kept[k].lat_min), float(kept[k].lat_max)],
        }
        for k in range(K)
    ]
    return cell_records, final_lat, final_lon, assign


def main() -> None:
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    gc = cfg.get("geocells", {})
    max_per_cell = gc.get("max_per_cell", 500)
    min_per_cell = gc.get("min_per_cell", 30)
    min_cell_side = gc.get("min_cell_side", 0.5)
    tag = gc.get("tag", "default")

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    cells, clat, clon, train_assign = build_cells(
        train, max_per_cell, min_per_cell, min_cell_side)
    K = len(cells)
    print(f"Built K={K} geocells (max_per_cell={max_per_cell}, "
          f"min_per_cell={min_per_cell}, min_side={min_cell_side})")

    val_assign = assign_nearest(val["lat"].to_numpy(), val["lon"].to_numpy(), clat, clon)
    test_assign = assign_nearest(test["lat"].to_numpy(), test["lon"].to_numpy(), clat, clon)

    with open(data_dir / f"geocells_{tag}.json", "w") as f:
        json.dump({"K": K, "cells": cells,
                   "params": {"max_per_cell": max_per_cell,
                              "min_per_cell": min_per_cell,
                              "min_cell_side": min_cell_side}}, f, indent=2)
    np.save(data_dir / f"cell_assign_train_{tag}.npy", train_assign)
    np.save(data_dir / f"cell_assign_val_{tag}.npy", val_assign)
    np.save(data_dir / f"cell_assign_test_{tag}.npy", test_assign)
    print(f"Saved geocells_{tag}.json + assignments. "
          f"Median cell count: {np.median([c['count'] for c in cells]):.0f}")


if __name__ == "__main__":
    main()
