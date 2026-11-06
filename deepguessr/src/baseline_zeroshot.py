"""Baseline A - Zero-shot CLIP.

Match cached test image embeddings to text prompts "a street view photo in
{country}." Predict argmax country. Predicted coordinates = mean train location
of that country. Reports single-prompt and prompt-ensemble results, plus a
top-20 (gt -> pred) confusion summary.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import open_clip
import pycountry

from common import get_device, load_config, save_run
from geo import mean_latlon
from metrics import evaluate, per_continent

# A few official names that CLIP understands better as common names.
NAME_OVERRIDES = {
    "US": "United States", "GB": "United Kingdom", "KR": "South Korea",
    "KP": "North Korea", "RU": "Russia", "IR": "Iran", "SY": "Syria",
    "VN": "Vietnam", "LA": "Laos", "TW": "Taiwan", "BO": "Bolivia",
    "VE": "Venezuela", "TZ": "Tanzania", "CZ": "Czech Republic",
    "MD": "Moldova", "BN": "Brunei",
}


def country_name(cc: str) -> str:
    if cc in NAME_OVERRIDES:
        return NAME_OVERRIDES[cc]
    try:
        c = pycountry.countries.get(alpha_2=cc)
        return getattr(c, "common_name", None) or c.name
    except Exception:
        return cc


def text_embeddings(model, tokenizer, names, templates, device):
    """Return (num_countries, dim) normalized text embeddings per template set
    (averaged over templates = ensemble)."""
    embs = []
    with torch.no_grad():
        for tmpl in templates:
            prompts = [tmpl.format(country=n) for n in names]
            toks = tokenizer(prompts).to(device)
            t = model.encode_text(toks)
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embs.append(t)
    return embs  # list of (C, D)


def run_variant(name, img_emb, text_emb, countries, country_centers,
                test, cfg, tag):
    # cosine sim = img @ text.T  (both normalized)
    sims = img_emb @ text_emb.T               # (N, C)
    pred_idx = sims.argmax(dim=1).cpu().numpy()
    pred_country = np.array([countries[i] for i in pred_idx])
    pred_lat = np.array([country_centers[c][0] for c in pred_country])
    pred_lon = np.array([country_centers[c][1] for c in pred_country])

    gt_lat = test["lat"].to_numpy()
    gt_lon = test["lon"].to_numpy()
    gt_country = test["country_code"].to_numpy()

    m = evaluate(pred_lat, pred_lon, gt_lat, gt_lon,
                 pred_country=pred_country, gt_country=gt_country)
    print(f"[{tag}] median_km={m['median_km']:.0f} "
          f"acc@200={m['acc@200km']:.3f} country_acc={m['country_acc']:.3f}")

    # top-20 confusions (gt -> pred), only wrong ones.
    conf = Counter(
        (g, p) for g, p in zip(gt_country, pred_country) if g != p
    ).most_common(20)
    preds = pd.DataFrame({
        "id": test["id"], "gt_lat": gt_lat, "gt_lon": gt_lon,
        "pred_lat": pred_lat, "pred_lon": pred_lon,
        "gt_country": gt_country, "pred_country": pred_country,
    })
    save_run(cfg, tag, m, preds, extra={
        "per_continent": per_continent(
            pred_lat, pred_lon, gt_lat, gt_lon,
            pred_country=pred_country, gt_country=gt_country),
        "top20_confusions": [[list(k), v] for k, v in conf],
        "n_candidate_countries": len(countries),
    })
    return m


def main() -> None:
    cfg = load_config()
    device = get_device(cfg["device"])
    data_dir = Path(cfg["paths"]["data_dir"])

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Candidate countries = those present in train. Representative point =
    # mean train location per country.
    countries = sorted(train["country_code"].unique().tolist())
    country_centers = {}
    for c in countries:
        sub = train[train["country_code"] == c]
        country_centers[c] = mean_latlon(sub["lat"].to_numpy(), sub["lon"].to_numpy())
    names = [country_name(c) for c in countries]
    print(f"{len(countries)} candidate countries")

    # Load cached test embeddings (ordered like test.csv).
    test_emb = np.load(data_dir / "emb_test.npy").astype(np.float32)
    test_ids = np.load(data_dir / "emb_test_ids.npy")
    assert np.array_equal(test_ids, test["id"].to_numpy()), "emb/test id mismatch"
    img_emb = torch.from_numpy(test_emb).to(device)

    model, _, _ = open_clip.create_model_and_transforms(
        cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"])
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["clip"]["model"])

    templates = cfg["zeroshot"]["prompt_templates"]
    text_embs = text_embeddings(model, tokenizer, names, templates, device)

    # Single prompt = first template.
    run_variant("single", img_emb, text_embs[0], countries, country_centers,
                test, cfg, "zeroshot_single")

    # Ensemble = mean of normalized text embeddings across templates, renormalized.
    ens = torch.stack(text_embs, dim=0).mean(dim=0)
    ens = ens / ens.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    run_variant("ensemble", img_emb, ens, countries, country_centers,
                test, cfg, "zeroshot_ensemble")


if __name__ == "__main__":
    main()
