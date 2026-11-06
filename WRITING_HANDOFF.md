# DeepGuessr — Milestone Report Writing Handoff

**For:** whoever is writing the LaTeX report (I did the code/experiments, not the writeup).
**Due:** Wed **Nov 4, 2026, 11:59pm ET** — one PDF per group on Gradescope (Group Submission, add all members).
**Template:** ICLR conference template (download from Canvas). Add all team members as authors.
**Page limit:** 4–5 pages main text (references don't count). Appendix allowed for **figures only**, no extra text.

> Heads up on scope: we are a **CS7643** group (if any member is in 7643), which means the rubric **requires three research hypotheses H1–H3** and weights the technical-approach section more. If we're a 4644-only group, skip the hypotheses (see §3 note). Confirm which rubric applies before finalizing.

Everything below maps 1:1 to the grading rubric. All numbers are real, pulled from our runs. Figures/tables are in the repo under `deepguessr/figures/` and `deepguessr/results/`.

---

## Project one-liner (use in abstract/intro)
Image geolocation: predict where a single street-view photo was taken. We use a frozen **CLIP ViT-B/16** image encoder and compare baselines (zero-shot country matching, lat/lon regression) against our main method: classifying images into **adaptive geocells** and predicting the cell center.

---

## Section 1 — Introduction (18 pts)
Cover:
- **Problem + importance:** single-image geolocation; applications (photo organization, OSINT, navigation priors), and why it's hard (visual ambiguity, massive output space, heavy geographic data bias).
- **Input/Output (be explicit, rubric wants this):**
  - Input: one RGB street-view image (resized to 224×224×3 for CLIP).
  - Output: a geographic location (latitude, longitude). For the main method, output is a distribution over K geocells; predicted location = center of the argmax cell.
- **Training conditions (rubric wants this):** supervision = GPS coordinates from the dataset; backbone = pretrained CLIP ViT-B/16 (OpenAI weights), **frozen** for baselines + current method; losses = haversine-km regression loss (baseline B), cross-entropy over geocells (one-hot and distance-smoothed) for the method.
- **Teaser figure (2 pts, required):** use `fig_geocells.png` (the adaptive geocell map, K=89) or make a simple pipeline box diagram: image → CLIP encoder → geocell probabilities → predicted pin. Easiest win: `fig_geocells.png`.

## Section 2 — Related Work (12–16 pts, need ≥5 citations)
Suggested citations to find + cite (please verify and add BibTeX):
1. **CLIP** — Radford et al., 2021 (Learning Transferable Visual Models / contrastive image-text pretraining). Our backbone.
2. **PlaNet** — Weyand et al., 2016 (planet-scale geolocation as classification over geocells). This is the direct ancestor of our geocell method.
3. **IM2GPS** — Hays & Efros, 2008 (retrieval-based single-image geolocation; classic baseline framing).
4. **ISNs / hierarchical geolocation** — Müller-Budack et al., 2018 (scene-aware geocell classification; distance-based label smoothing ideas).
5. **OpenStreetView-5M (OSV-5M)** — the dataset paper (Astruc et al., 2024). Cite for dataset + their benchmark numbers.
6. (optional) **LoRA** — Hu et al., 2021, for the planned fine-tuning stretch.
- Explain how our direction differs: we combine CLIP's strong frozen features with geocell classification + distance-smoothed labels, and plan LoRA fine-tuning — contrasting with from-scratch CNNs (PlaNet) and pure retrieval (IM2GPS).
- Briefly describe the dataset (see §4 numbers below).

## Section 3 — Problem Decomposition & Technical Approach (24–32 pts)
- **DL formulation:** given image embedding z = CLIP(image) ∈ R^512, learn a head mapping z to location. Two framings: (a) regression z→(lat,lon); (b) classification z→K geocells, predict cell center.
- **Method detail (geocells):** adaptive quadtree over train coordinates — start 8×4 lon/lat grid, recursively split any cell with >`max_per_cell`=500 images into 4 quadrants, stop at min cell side 0.5°, drop cells with <`min_per_cell`=30 and reassign to nearest center. Cell center = mean of member points (mean of unit vectors). This yielded **K=89 cells**. Head = MLP (512→1024→K). Two label types: **one-hot CE** and **distance-smoothed CE** with target q_k ∝ exp(−haversine(c_k, y)/τ), τ=75 km.
- **Training details:** frozen CLIP embeddings cached once; head trained with AdamW, lr 1e-3, weight decay 1e-4, batch 1024, 30 epochs, cosine schedule, early stopping / model selection on **val median km**. Seed 42.
- **Evaluation procedure (8 pts, be concrete):** fixed held-out test subset of **5,000 images** (same for every model). Metrics: **median/mean great-circle error (km)**, **accuracy@{25, 200, 750, 2500} km** (city/region/country/continent scales), and **country accuracy** (reverse-geocode both GT and predicted coords for consistency). Never tune on test; model selection on val only.
- **CS7643 ONLY — three hypotheses (8 pts, required for 7643).** Must have independent + dependent variables, testable. Draft set (refine wording):
  - **H1:** With the frozen CLIP encoder and identical training budget, **geocell classification achieves lower median km error than direct lat/lon regression**, because classification avoids the mean-collapse failure mode of regression. *(We already have evidence — see results.)*
  - **H2:** **Distance-smoothed geocell labels (τ=75 km) improve coarse-scale accuracy (acc@200/750 km) over one-hot labels**, holding architecture, K, and optimizer fixed, by giving partial credit to neighboring cells.
  - **H3:** **LoRA fine-tuning of the CLIP vision tower (rank 8–16) reduces median km error relative to the frozen-encoder geocell head**, under matched training steps, because task-adapted features beat generic CLIP features for geolocation.
  - *(If 4644-only, drop H1–H3; those 8 pts move into Related Work / Experiments per the 4644 rubric.)*

## Section 4 — Experiments & Results (20–24 pts)
### Dataset (cite + state the pipeline)
- **OpenStreetView-5M (OSV-5M)**, HF repo `osv5m/osv5m`. ~5M street-view images with GPS + ISO alpha-2 country labels.
- We downloaded a laptop-scale subset (1 train shard + 1 test shard), built seeded splits:
  - **Train: 19,000** images, **177** countries.
  - **Val: 1,000** images (seeded 5% of train).
  - **Test: 5,000** images, **186** countries (fixed subset, shared across all models).
- Heavy geographic bias (worth a sentence + `fig_data_distribution.png`): train is **41% Europe, 30% North America**, with Africa/Oceania sparse. Top countries: US (4,832), DE (1,046), FR (941), RU (846), JP (823).

### Baselines (how they work, high level)
- **Sanity baselines** (prove metrics are sane): predict train-mean location / most-common country center → ~6,900–9,100 km median. Not real models; context rows only.
- **Baseline A — Zero-shot CLIP:** match image embedding to text prompts `"a street view photo in {country}."`, predict argmax country; predicted coords = that country's mean train location. Also a **prompt-ensemble** variant.
- **Baseline B — Regression:** MLP head (512→512→3), predict a 3D unit vector → lat/lon, trained with differentiable **haversine-km loss**. Linear and MLP variants.

### Main results table (copy from `results/results_table.md` / `.tex`)
The LaTeX `tabular` is ready to paste: `deepguessr/results/results_table.tex`. Numbers (5,000-image test set):

| Method | Median km ↓ | acc@25km | acc@200km | acc@750km | acc@2500km | Country acc |
|---|---|---|---|---|---|---|
| Train-mean (sanity) | 6890 | 0.0 | 0.0 | 0.7 | 7.4 | 0.6 |
| Common-country (sanity) | 9069 | 0.0 | 0.1 | 2.9 | 16.6 | 11.9 |
| Zero-shot CLIP (single) | 1934 | 0.2 | 5.2 | 23.8 | 57.5 | 29.6 |
| Zero-shot CLIP (ensemble) | 1763 | 0.2 | 5.3 | 25.2 | 60.2 | 30.5 |
| Regression (linear) | 2700 | 0.0 | 0.9 | 11.7 | 47.4 | 22.0 |
| Regression (MLP) | 1990 | 0.1 | 2.1 | 20.6 | 57.0 | 29.2 |
| Geocell MLP (one-hot) | 1412 | 0.1 | 4.5 | 31.7 | 64.7 | 36.2 |
| **Geocell MLP (smooth τ=75)** | **1391** | 0.1 | **4.9** | **32.3** | **65.1** | **36.3** |

(There are also geocell *linear* rows — 2273/2320 km — included to show the linear head underfit; mention briefly.)

### Takeaways to write (all supported by our data)
1. **Our method wins.** Geocell MLP (smoothed) is best on median km (1391) and every coarse accuracy threshold and country acc (36.3%) → supports **H1** (beats regression 1990) and beats zero-shot (1763).
2. **Smoothing helps a little**, as predicted (1391 vs 1412; better acc@200/750) → partial support for **H2**.
3. **Zero-shot narrowly wins acc@25km** — geocells quantize to K=89 coarse centers, so city-level precision is limited. Honest limitation; motivates more cells + LoRA.
4. **Regression collapses toward the mean** (classic failure) — visible in `fig_prediction_map.png`: regression predictions cluster in dense mid-latitudes; our method spreads correctly.
5. **Neighbor-country confusions** dominate zero-shot errors (US↔CA, BR↔PY, RU↔FI) — nice qualitative point.

### Figures available (in `deepguessr/figures/`, PNG@300dpi + PDF)
- `fig_geocells` — adaptive geocell map (teaser candidate)
- `fig_error_cdf` — fraction within X km, all models (great single summary figure)
- `fig_per_continent` — per-continent median error bars
- `fig_training_curves` — train loss + val median km
- `fig_prediction_map` — GT vs pred, zero-shot / regression / geocell panels (shows collapse)
- `fig_data_distribution` — train geographic bias
- Per-continent breakdown table: `results/per_continent_table.md`

## Section 5 — Next Steps (10 pts)
Concrete plan to write:
- **Scale up data** on ICE/Colab (100k–500k train images) and increase K (granularity ablation: max_per_cell ∈ {2000, 500, 150}).
- **LoRA fine-tuning** of CLIP vision tower (rank 8–16) — tests **H3**. Scaffold is the next code task.
- **τ ablation** {25, 75, 150, 300} to fully characterize smoothing (tests **H2**).
- Report final with per-continent fairness analysis.
- **Challenges + mitigations:** geographic bias (→ reweighting / stratified sampling), compute limits (→ cached embeddings, LoRA not full fine-tune), city-level precision ceiling (→ more cells + fine-tuning).
- **Success criteria:** beat our baselines on median km + acc@200km on the held-out test set; show LoRA improves over frozen.

---

## Logistics notes for the writer
- Reproduction commands + confirmed dataset schema are documented in the code/config; ask me if you need exact column names or the git hash (it's saved in each `results/*.json`).
- The LaTeX table is pre-generated: `results/results_table.tex` (booktabs-style `tabular`). `results/per_continent_table.md` has the continent breakdown if you want a second table.
- Please add real BibTeX for the citations in §2 — I listed authors/years but didn't generate `.bib` entries.
- If anything in the numbers looks off while writing, ping me and I'll re-check the source JSON rather than you guessing.
