"""Aggregate results/*.json into a results table (markdown + LaTeX) and a
per-continent table (markdown). Bold the best value per column.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import load_config

# Row order + display names. Keyed by run_name (json stem).
ROW_ORDER = [
    ("sanity_train_mean", "Train-mean (sanity)"),
    ("sanity_common_country", "Common-country (sanity)"),
    ("zeroshot_single", "Zero-shot CLIP (single)"),
    ("zeroshot_ensemble", "Zero-shot CLIP (ensemble)"),
    ("regression_linear", "Regression (linear)"),
    ("regression_mlp", "Regression (MLP)"),
    ("geocell_linear_onehot", "Geocell linear (one-hot)"),
    ("geocell_linear_smooth_tau75", "Geocell linear (smooth tau=75)"),
    ("geocell_mlp_onehot", "Geocell MLP (one-hot)"),
    ("geocell_mlp_smooth_tau75", "Geocell MLP (smooth tau=75)"),
    ("lora_geocell_smooth_tau75", "LoRA geocell MLP (smooth tau=75)"),
]

# (metric key, header, kind) where kind in {"km","pct"} and arrow direction.
COLS = [
    ("median_km", "Median km", "km", "down"),
    ("acc@25km", "25 km", "pct", "up"),
    ("acc@200km", "200 km", "pct", "up"),
    ("acc@750km", "750 km", "pct", "up"),
    ("acc@2500km", "2500 km", "pct", "up"),
    ("country_acc", "Country acc", "pct", "up"),
]


def load_metrics(results_dir: Path) -> dict:
    out = {}
    for key, _ in ROW_ORDER:
        p = results_dir / f"{key}.json"
        if p.exists():
            out[key] = json.load(open(p))
    return out


def fmt(val: float, kind: str) -> str:
    return f"{val:.0f}" if kind == "km" else f"{val * 100:.1f}"


def best_index(data: dict, metric: str, direction: str) -> str | None:
    vals = {k: d["metrics"][metric] for k, d in data.items()}
    if not vals:
        return None
    return (min if direction == "down" else max)(vals, key=vals.get)


def build_markdown(data: dict) -> str:
    header = "| Method | " + " | ".join(
        f"{h} {'↓' if dr == 'down' else '↑'}" for _, h, _, dr in COLS
    ) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    lines = [header, sep]
    best = {m: best_index(data, m, dr) for m, _, _, dr in COLS}
    for key, name in ROW_ORDER:
        if key not in data:
            continue
        cells = [name]
        for m, _, kind, _ in COLS:
            v = fmt(data[key]["metrics"][m], kind)
            if best[m] == key:
                v = f"**{v}**"
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_latex(data: dict) -> str:
    best = {m: best_index(data, m, dr) for m, _, _, dr in COLS}
    col_spec = "l" + "r" * len(COLS)
    arrows = [f"{h} $\\{'downarrow' if dr == 'down' else 'uparrow'}$"
              for _, h, _, dr in COLS]
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Method & " + " & ".join(arrows) + " \\\\",
        "\\midrule",
    ]
    for key, name in ROW_ORDER:
        if key not in data:
            continue
        cells = [name]
        for m, _, kind, _ in COLS:
            v = fmt(data[key]["metrics"][m], kind)
            if best[m] == key:
                v = "\\textbf{" + v + "}"
            cells.append(v)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_per_continent(data: dict) -> str:
    # Columns = continents (union), two sub-metrics per model row.
    conts = []
    for d in data.values():
        for c in d.get("per_continent", {}):
            if c not in conts:
                conts.append(c)
    conts = sorted(conts)
    lines = ["## Per-continent median km (↓)\n"]
    lines.append("| Method | " + " | ".join(conts) + " |")
    lines.append("|" + "---|" * (len(conts) + 1))
    for key, name in ROW_ORDER:
        if key not in data or "per_continent" not in data[key]:
            continue
        pc = data[key]["per_continent"]
        cells = [name] + [
            f"{pc[c]['median_km']:.0f}" if c in pc else "-" for c in conts
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Per-continent country accuracy (↑)\n")
    lines.append("| Method | " + " | ".join(conts) + " |")
    lines.append("|" + "---|" * (len(conts) + 1))
    for key, name in ROW_ORDER:
        if key not in data or "per_continent" not in data[key]:
            continue
        pc = data[key]["per_continent"]
        cells = [name] + [
            f"{pc[c]['country_acc'] * 100:.1f}" if c in pc else "-" for c in conts
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    data = load_metrics(results_dir)
    if not data:
        print("No result JSONs found.")
        return

    (results_dir / "results_table.md").write_text(build_markdown(data))
    (results_dir / "results_table.tex").write_text(build_latex(data))
    (results_dir / "per_continent_table.md").write_text(build_per_continent(data))
    print("Wrote:")
    for f in ("results_table.md", "results_table.tex", "per_continent_table.md"):
        print("  ", results_dir / f)
    print("\n" + build_markdown(data))


if __name__ == "__main__":
    main()
