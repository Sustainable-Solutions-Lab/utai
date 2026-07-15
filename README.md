# UTAI — T2 dose-response analysis

Regression of near-surface air temperature (`T2`) against aerosol/agent **release
rate**, from a set of WRF-style ensemble simulations. For each hour of a multi-day
episode and each evaluation area we quantify how much the release cools (or warms)
the surface and how that response scales with the release rate.

Everything is done **separately** for each of the four `(episode, area)`
combinations, and independently for each hour.

## Data

Input: `data/input/T2_summary.csv` (5808 rows) — one temperature `value` per
combination of:

| column            | meaning |
|-------------------|---------|
| `episode`         | case date `yymmdd` — `240527`, `240727` (2 episodes) |
| `scenario`        | release case: `ctl`, `1000_5x5`, `10000_5x5`, `100000_5x5` |
| `ens`             | ensemble member `e1`/`e2`/`e3` (3 realizations) |
| `evaluatino_area` | `city` or `region` (**misspelled upstream**; code renames to `area`) |
| `hour_index`      | hour 1–121 (~5 days) → `hour` |
| `value`           | `T2`, near-surface temperature in K (~297–319) → `T2` |

`2 episodes × 4 scenarios × 3 members × 2 areas × 121 hours = 5808`.

**Release rate** (kt/h) is derived from the scenario label: `ctl → 0`; otherwise
the number before the first `_`, divided by 1000. So the four doses are
**0, 1, 10, 100 kt/h**.

The ensemble members are **matched across scenarios**: `e1` is the same weather
realization in `ctl`, `1000_5x5`, `10000_5x5`, `100000_5x5`. This pairing is central
to the analysis (see below).

## The model

For each `(episode, area)` we model the **anomaly** relative to the control:

```
T2_release - T2_ctl  =  T2_scale(h) · f(release_rate)
```

- **`T2_scale(h)`** varies per hour. Both dose-response shapes are normalized so
  the factor `f(10) = 1`; hence `T2_scale` is exactly the perturbation at 10 kt/h
  (reported as `deltaT2_10`).
- **`f(release_rate)`** is the dose-response shape, `f(0) = 0` (no anomaly with no
  release), governed by one shape parameter held **constant per `(episode, area)`**.

Two shapes are implemented:

| shape | `f(release_rate)` | shape parameter |
|-------|-------------------|-----------------|
| power law | `(release_rate/10)**beta` | `beta` (sub-linear if `<1`) |
| exponential saturation | `(1 - e^(-rate/release_scale)) / (1 - e^(-10/release_scale))` | `release_scale` (kt/h) |

The **baseline** `T2_0(h)` (the `release_rate = 0` temperature) is taken directly
as the control ensemble mean ± its standard error — it is not a fitted intercept,
so the nonlinear curve cannot trade against it.

## Why the pairing matters

Each raw temperature decomposes as

```
T2(dose, member, hour) = mu(dose, hour) + m(member, hour) + eps
```

where `m(member, hour)` is a large member-specific meteorology offset shared by
**all** doses of that member (empirically the ctl-vs-r10 member correlation is
median **+0.93**). Fitting raw `T2` and treating the 12 samples per hour as
independent does **not** bias the estimates (the design is balanced and crossed, so
`m` is orthogonal to dose) but **inflates the standard errors**. Subtracting the
matched control forms the paired anomaly `d_{s,e}(h) = T2_{s,e}(h) - T2_{ctl,e}(h)`,
which removes `m` exactly. This is the recommended method.

## Fitting and error propagation

**Fit** — profiled (separable) least squares. For a fixed shape parameter the model
is linear in `T2_scale`, so each hour's value is the closed-form through-origin
slope; only the scalar shape parameter needs a 1-D search
(`scipy.optimize.minimize_scalar`) minimizing the total residual SS over all 121
hours. `release_scale` is searched in log space.

**Uncertainty** — the **ensemble member is the unit of replication**. Each member
supplies a complete paired data set, hence one independent through-origin slope
`u_e(h)`; therefore

```
T2_scale(h)            = mean_e u_e(h)
SE[T2_scale(h) | shape] = sd(u_e, ddof=1) / sqrt(n_members)     (exact, distribution-free)
```

The shape parameter and the shape-inclusive SE of `T2_scale` come from a
**delete-one-member jackknife** (each whole member — its ctl *and* experiments —
removed together, preserving the pairing). All standard errors scale as
`~1/sqrt(n_members)`, so they tighten as ensemble members are added.

**Reading the plots** — shaded bands are **±1 standard error**. A 95% CI multiplies
the SE by Student-t with `n_members − 1` dof; at `n = 3` that factor is ≈ 4.3, so a
95% band is ~4× wider than drawn. The factor falls toward 1.96 (and the SEs shrink)
as members are added.

## Key findings

- **Baseline** `T2_0` is a clean 5-day diurnal cycle; **city warmer than region**
  (urban heat island), and **`240527` warmer than `240727`**.
- The dose-response is **sub-linear / saturating** in both episodes. The
  **exponential-saturation shape fits better** than the power law
  (median per-hour `r2_anom`):

  | episode | area | `release_scale` (kt/h) | r² (saturation) | r² (power) |
  |---------|------|------------------------|-----------------|------------|
  | 240527 | city | 3.3 | 0.61 | 0.31 |
  | 240527 | region | 23.9 | 0.57 | 0.53 |
  | 240727 | city | 1.8 | 0.06 | 0.03 |
  | 240727 | region | 8.8 | 0.78 | 0.75 |

  **City saturates fast** (small `release_scale`), **region is closer to linear**
  over the 1–100 kt/h range.
- At **100 kt/h** the response is predominantly cooling; **region** shows a steady,
  tight cooling signal while **city** is noisier and can flip sign day↔night.
- **Pairing halves the uncertainty** (paired SE ≈ 0.47× the unpaired SE), and the
  `region · 240527 @ 10 kt/h` mid-dose is genuinely harder to fit under either
  shape (not a curve-shape artifact).

## Scripts

All read `data/input/T2_summary.csv` and write to `data/output/` (git-ignored).
Run any script with the project virtualenv, e.g. `.venv/bin/python src/<script>.py`.

| script | purpose |
|--------|---------|
| `fit_t2_shared_beta_anomaly.py` | **Recommended.** Paired-anomaly fit, power-law shape, shared `beta`. |
| `fit_t2_saturation_anomaly.py` | Paired-anomaly fit, exponential-saturation shape, shared `release_scale`. |
| `ctl_anomaly.py` | Model-free control means and paired release-minus-ctl anomalies (sanity check). |
| `scatter_pred_obs.py` | Predicted-vs-observed anomaly, parametric in hour (12 panels). Arg: `power` (default) or `saturation`. |
| `overlay_r10.py` | Overlays the raw r10 anomaly on the fitted `deltaT2_10` (cross-check). |
| `fit_t2.py` | Earlier method: per-hour fit with free `beta` (kept for comparison). |
| `fit_t2_shared_beta.py` | Earlier method: shared `beta` on the raw (unpaired) `T2` (kept for comparison). |

Order: run `fit_t2_shared_beta_anomaly.py` and `fit_t2_saturation_anomaly.py` first;
`scatter_pred_obs.py` reads their CSVs, and `overlay_r10.py` reads
`fit_t2_shared_beta.py` + `ctl_anomaly.py` outputs.

## Plot conventions

Line plots encode **line style = area** (dashed city / solid region) and
**color = episode** (blue `240527` / orange `240727`); bar charts color by episode
with city hatched / region solid. Hour axes tick every 24 h (day boundaries) with
12 h minor gridlines.

## Outputs (`data/output/`)

Per method: a per-hour `*_fit.csv`/`.xlsx` (with `T2_0`, `deltaT2_10` and their
standard errors, the shape parameter, `r2_anom`), a shape-parameter summary CSV,
and coefficient plots (`ctl_baseline.png`, `deltaT2_10_*.png`, the shape-parameter
bar chart, and the predicted-vs-observed scatter). See `CLAUDE.md` for the exact
file names per script.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy statsmodels scikit-learn matplotlib openpyxl
.venv/bin/python src/fit_t2_shared_beta_anomaly.py
```
