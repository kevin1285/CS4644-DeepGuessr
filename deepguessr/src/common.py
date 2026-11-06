"""Shared helpers: config loading, seeding, device selection, paths."""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "deepguessr" / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve path entries to absolute Paths rooted at the repo.
    for k, v in cfg.get("paths", {}).items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else (REPO_ROOT / p)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def get_device(requested: str = "mps") -> str:
    import torch

    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for p in cfg.get("paths", {}).values():
        Path(p).mkdir(parents=True, exist_ok=True)


def save_run(cfg: dict[str, Any], run_name: str, metrics: dict, preds_df=None,
             extra: dict | None = None) -> None:
    """Write results/<run_name>.json (metrics+config+seed+git hash) and
    optionally results/<run_name>_preds.csv."""
    import json

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "metrics": metrics,
        "seed": cfg.get("seed"),
        "git_hash": git_hash(),
        "clip": cfg.get("clip"),
    }
    if extra:
        payload.update(extra)
    with open(results_dir / f"{run_name}.json", "w") as f:
        json.dump(payload, f, indent=2)
    if preds_df is not None:
        preds_df.to_csv(results_dir / f"{run_name}_preds.csv", index=False)
    print(f"[{run_name}] metrics -> {results_dir / (run_name + '.json')}")
