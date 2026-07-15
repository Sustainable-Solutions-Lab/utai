# CLAUDE.md — working notes for this repo

Project overview, the model, the statistics, and findings live in **`README.md`** —
read it first. This file is the operational cheat-sheet for making changes.

## Environment

- Use the project virtualenv interpreter directly: **`.venv/bin/python`** (packages
  are installed there, not in the system Python). Activation is optional.
- Installed: numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, openpyxl.

## Data quirks (do not "fix" the source)

- The input column `evaluatino_area` is **misspelled upstream**; `fit_t2.load_data`
  renames it to `area` and renames `value → T2`, `hour_index → hour`. Leave the CSV
  as-is.
- `release_rate` (kt/h) is derived in `fit_t2.release_rate`: `ctl → 0`, else the
  number before the first `_` divided by 1000 → `{0, 1, 10, 100}`.
- `episode` is loaded as a **string** (`'240527'`). When you read a generated CSV
  back and merge against fresh data, cast `df['episode'] = df['episode'].astype(str)`
  first, or the merge fails on int-vs-object.

## Conventions

- All generated files go under **`data/output/`** (git-ignored, along with `.venv/`,
  `*.xlsx`, `*.png`).
- Shared plotting helpers live in `fit_t2.py`: `plot_coefficient` (line + optional
  ±z·SE band; `day_ticks` puts major ticks every 24 h / minor 12 h) and, in
  `fit_t2_shared_beta.py`, `plot_beta_bars` (shared-parameter bar chart).
- Plot encoding: **line style = area** (dashed city / solid region), **color =
  episode** (blue `240527` / orange `240727`); bars color by episode, city hatched.
  Bands are ±1 SE unless a script overrides `z`.
- The paired-anomaly scripts share `build_anomalies` and `sem` from
  `fit_t2_shared_beta_anomaly.py`.

## Scripts and their outputs (all under `data/output/`)

- **`fit_t2_shared_beta_anomaly.py`** (recommended, power-law shape) →
  `t2_anomaly_fit.csv`/`.xlsx`, `t2_anomaly_beta.csv`, `ctl_baseline.png`,
  `deltaT2_10_anomaly.png`, `beta_anomaly.png`.
- **`fit_t2_saturation_anomaly.py`** (exp-saturation shape) →
  `t2_saturation_fit.csv`/`.xlsx`, `t2_saturation_scale.csv`,
  `deltaT2_10_saturation.png`, `release_scale_saturation.png`.
- **`scatter_pred_obs.py [power|saturation]`** → `scatter_pred_vs_obs_power.png` /
  `scatter_pred_vs_obs_saturation.png` (suffixed by model; reads the matching
  `*_fit.csv`).
- **`ctl_anomaly.py`** → `ctl_mean_se.csv`, `t2_anomalies.csv`/`.xlsx`,
  `ctl_mean_se.png`, `anomaly_r1/r10/r100.png`.
- **`overlay_r10.py`** → `deltaT2_10_vs_anomaly_r10.png`, `r10_vs_deltaT2_10.csv`
  (reads `fit_t2_shared_beta.py` + `ctl_anomaly.py` outputs).
- **`plot_member_anomalies.py [episode] [area] [release_rate] [model]`** (default
  `240527 region 10 power`) → `member_anomalies_<episode>_<area>_r<rate>_<model>.png`:
  the three per-member `T2_exp - T2_ctl` traces, their mean, and the model
  **prediction at that release rate** overlaid (`T2_scale(h) * dose_factor(rate)`;
  factor = 1 at 10 kt/h).
- **`fit_t2.py`** (earlier, free-`beta` per hour) → `t2_hourly_fits.csv`/`.xlsx`,
  `T2_0.png`, `deltaT2_10.png`, `beta.png`.
- **`fit_t2_shared_beta.py`** (earlier, shared `beta` on raw `T2`) →
  `t2_sharedbeta_fits.csv`/`.xlsx`, `t2_sharedbeta_beta.csv`,
  `T2_0_sharedbeta.png`, `deltaT2_10_sharedbeta.png`, `beta_sharedbeta.png`.

## Layout

```
data/input/T2_summary.csv   source data (read-only)
data/output/                generated CSV/XLSX/PNG (git-ignored)
src/*.py                    analysis scripts (see above)
.venv/                      project virtualenv (git-ignored)
```
