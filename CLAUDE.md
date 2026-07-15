# UTAI — T2 dose-response regression

Hourly regression of near-surface temperature (`T2`) against aerosol/agent
**release rate**, fit independently for each hour of a multi-day episode and for
each city/region evaluation area.

## Data

Input: `data/input/T2_summary.csv` (5808 rows). One `value` per combination of:

| column              | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| `episode`           | case date as `yymmdd` (`240527`, `240727`) — 2 episodes        |
| `scenario`          | release case: `ctl`, `1000_5x5`, `10000_5x5`, `100000_5x5`     |
| `ens`               | ensemble member `e1`/`e2`/`e3` (3 stochastic realizations)     |
| `evaluatino_area`   | `city` or `region` (**note the source spelling**; normalized to `area`) |
| `hour_index`        | hour 1–121 (~5 days) → used as `hour`                          |
| `value`             | `T2`, near-surface temperature in K (~297–319) → used as `T2`  |

**release_rate (kt/h)** is derived from `scenario`: `ctl` → 0; otherwise the
number before the first `_`, divided by 1000. So `1000_5x5`→1, `10000_5x5`→10,
`100000_5x5`→100. Values are therefore **0, 1, 10, 100**.

## The four analyses

Everything is done **separately** for each of the 4 `(episode, area)` combinations:
`(240527, city)`, `(240527, region)`, `(240727, city)`, `(240727, region)`.

The scripts implement three progressively better treatments of the same model.
`fit_t2.py` (per-hour, free `beta`) and `fit_t2_shared_beta.py` (shared `beta`,
raw fit) are kept for comparison, but the **recommended** method is the
paired-anomaly fit below — it is the statistically correct treatment of the
matched ensemble design.

## Recommended method — paired-anomaly shared-beta fit (`fit_t2_shared_beta_anomaly.py`)

### Why paired

The three ensemble members `e1/e2/e3` are the **same weather realization** across
scenarios (matched initial conditions; only the release differs). So each raw
temperature decomposes as

```
T2(dose, member, hour) = mu(dose, hour) + m(member, hour) + eps
```

where `m(member, hour)` is a large member-specific meteorology offset shared by
**all** doses of that member at that hour (empirically the ctl-vs-r10 member
correlation has median +0.93). Fitting raw `T2` and treating the 12 samples as
independent (the earlier scripts) leaves `m` in the residual, which does **not**
bias `deltaT2_10`/`beta` (the design is balanced and crossed, so `m` is orthogonal
to dose) but **inflates their standard errors**. Subtracting the matched control
removes `m` exactly.

Note `m(member, hour)` is estimated *per hour*, not once: because every hour is fit
independently, the member offset is free to change hour to hour — correct, since the
ensemble members diverge chaotically over the 5-day run.

### Model

Per `(episode, area)`:

- **Baseline** (the `release_rate = 0` value, i.e. `T2_0` by definition):
  ```
  T2_0(h)     = mean over ctl members of T2(h)
  SE[T2_0(h)] = sd_ctl(h) / sqrt(n_ctl)
  ```
  This is reported/plotted as the control mean +/- SE (it is not a fitted
  intercept — pinning the baseline to the control keeps the nonlinear curve from
  trading against it).

- **Paired anomaly** for each release member (`s` = dose in {1,10,100} kt/h):
  ```
  d_{s,e}(h) = T2_{s,e}(h) - T2_{ctl,e}(h)          (matched by member e)
  ```
  fit **through the origin** (no anomaly at `release_rate = 0`):
  ```
  d = deltaT2_10(h) * (release_rate / 10) ** beta
  ```
  with `beta` **constant per (episode, area)** and `deltaT2_10` free per hour.

### Fitting — profiled (separable) least squares

For a fixed `beta`, set `x_s = (r_s/10)**beta`; the model is linear in
`deltaT2_10`, so each hour's slope is closed-form through the origin,
`deltaT2_10(h) = sum(x_s d_{s,e}) / sum(x_s^2)`. Only the scalar `beta` needs a
search — `scipy.optimize.minimize_scalar` (bounded `[0.01, 10]`) minimizing the
total residual SS over all 121 hours. This is the "nonlinear fit for beta wrapping
a linear regression" structure.

### Error propagation — the ensemble member is the unit of replication

Each member `e` supplies a *complete* paired data set, hence one independent
through-origin slope
```
u_e(h) = sum_s x_s d_{s,e}(h) / sum_s x_s^2 .
```
Therefore
```
deltaT2_10(h)          = mean_e u_e(h)                       (= pooled slope)
SE[deltaT2_10(h) | beta] = sd(u_e, ddof=1) / sqrt(n_members)
```
This conditional SE is **exact and distribution-free**: it needs no
homoscedasticity or additivity assumption, and it automatically carries the
shared-control correlation because the whole difference lives inside each `u_e`.

`beta` and the **beta-inclusive** SE of `deltaT2_10` come from a
**delete-one-member jackknife**: refit with each whole member removed (its ctl
*and* its experiments drop together, so the pairing is preserved), giving
`beta_(-e)` and `deltaT2_10_(-e)(h)`, then
```
SE_jk = sqrt( (n-1)/n * sum_e ( theta_(-e) - mean_e theta_(-e) )^2 ).
```
The jackknife total SE of `deltaT2_10(h)` includes `beta`'s contribution; comparing
it to the conditional SE shows how much `beta` uncertainty adds (here: little,
because `beta` is pooled over 121 hours and tightly pinned).

All SEs scale as `~1/sqrt(n_members)`, so they tighten as ensemble members are
added — the reason this framework is worth getting right now with `n = 3`.

### Reading the bands (small-n note)

Plot bands are **+/- 1 standard error** (`z = 1`). A 95% confidence interval
multiplies the SE by Student-t with `n_members - 1` dof; with `n = 3` that factor
is `t_{2, 0.975} ≈ 4.30` (and the jackknife itself has only 2 dof), so a 95% band
is ~4x wider than what is drawn. As members are added the multiplier falls toward
1.96 and the SEs themselves shrink.

### Outputs (`data/output/`)

- `t2_anomaly_fit.csv` / `.xlsx` — per `(episode, area, hour)`: `T2_0`, `T2_0_se`,
  `deltaT2_10`, `deltaT2_10_se_cond` (conditional on `beta`),
  `deltaT2_10_se_total` (jackknife, incl. `beta`), `beta`, `r2_anom`, member/ctl
  counts. The `.xlsx` has a second `beta` sheet; `t2_anomaly_beta.csv` is that
  summary (`beta`, jackknife `beta_se`).
- `ctl_baseline.png` — baseline `T2_0` = control mean +/- 1 SE vs. hour.
- `deltaT2_10_anomaly.png` — `deltaT2_10` vs. hour, +/- 1 SE band (incl. `beta`).
- `beta_anomaly.png` — the four `beta` with jackknife-SE whiskers.

Fitted `beta` (paired anomaly): city `0.183`/`0.159`, region `0.486`/`0.283`
(240527/240727) — all sub-linear (saturating dose-response).

### Predicted-vs-observed diagnostic — `src/scatter_pred_obs.py`

A 12-panel page (rows = the 4 `(episode, area)` combos, cols = release rate
1/10/100 kt/h) plotting, **parametric in hour**, the member-mean observed anomaly
`mean_e(T2_exp - T2_ctl)(h)` on x against the fitted prediction
`deltaT2_10(h)*(rate/10)**beta` on y, colored by hour and joined in hour order,
with a 1:1 reference line and per-panel RMSE / correlation. 100 kt/h hugs 1:1
(r ~0.98-1.00); 1 kt/h is noisy (tiny signal); `region 240527 @ 10 kt/h` bows off
1:1 (the mid dose the single shared `beta` fits worst). Reads `t2_anomaly_fit.csv`;
output `scatter_pred_vs_obs.png`.

### Alternative dose-response shape — exponential saturation (`fit_t2_saturation_anomaly.py`)

Same paired-anomaly framework and error propagation, different curve. Instead of
the power law it uses an exponential-saturation shape:

```
deltaT2 = T2_scale * (1 - exp(-release_rate/release_scale))
                   / (1 - exp(-10/release_scale))
```

- factor = 0 at `release_rate = 0` (fit through the origin) and = 1 at
  `release_rate = 10`, so `T2_scale(h)` is again the perturbation at 10 kt/h —
  kept in the `deltaT2_10` column so the two forms are directly comparable;
- `release_scale` (kt/h, one constant per `(episode, area)`) is the curvature
  knob, playing `beta`'s role: `release_scale -> infinity` gives a linear dose
  response, `release_scale -> 0` saturates immediately. It is the release rate at
  which the response reaches `1 - 1/e` (~63%) of its far-field value.

Fitting is identical profiled least squares — `release_scale` searched in log
space (`[0.05, 2e4]` kt/h) with the same member-replicate + jackknife SEs.
`(1 - exp(-x))` is evaluated with `-expm1(-x)` to avoid cancellation at large
`release_scale`. The dose factor is 0 at `rate = 0`.

This shape fits **better** than the power law (median per-hour `r2_anom`):

| episode | area   | release_scale (kt/h) | r2 (sat.) | r2 (power) |
|---------|--------|----------------------|-----------|------------|
| 240527  | city   | 3.3                  | 0.61      | 0.31       |
| 240527  | region | 23.9                 | 0.57      | 0.53       |
| 240727  | city   | 1.8                  | 0.06      | 0.03       |
| 240727  | region | 8.8                  | 0.78      | 0.75       |

City saturates fast (small `release_scale`); region is closer to linear over the
1-100 kt/h range (large `release_scale`, and `240527 region` is only weakly
constrained — large jackknife SE).

Outputs (`data/output/`): `t2_saturation_fit.csv` / `.xlsx` (per-hour `deltaT2_10`
+ SEs and `release_scale`; second `release_scale` sheet), `t2_saturation_scale.csv`,
`deltaT2_10_saturation.png`, `release_scale_saturation.png`.

## Earlier methods (kept for comparison)

## Model

For each `(episode, area)` and **each hour `h` independently**, fit across the 12
samples (4 release cases × 3 ensemble members):

```
T2  =  T2_0  +  deltaT2_10 · (release_rate / 10) ** beta
```

- **`T2_0`** — control temperature. At `release_rate = 0` the power term is 0, so
  `T2_0` is pinned by the three control samples.
- **`deltaT2_10`** — temperature perturbation at 10 kt/h (the term equals `deltaT2_10`
  when `release_rate = 10`, since `(10/10)**beta = 1`).
- **`beta`** — dose-response curvature/exponent. `beta<1` sub-linear (saturating),
  `beta>1` super-linear, `beta=1` linear in dose.

Fit is nonlinear least squares (`scipy.optimize.curve_fit`, `trf`), with
`beta` bounded to `[0.01, 10]` and `T2_0`, `deltaT2_10` unbounded. The `rr=0`
term is pinned to exactly 0 to avoid `0**beta` issues.

### Known caveat — per-hour `beta` is weakly identified

`T2_0` is robust (control-pinned). In the per-hour fit `deltaT2_10` and `beta` are
**not** always well-constrained: with only 3 realizations per dose and a
frequently non-monotone response across doses, many hours have a near-zero
perturbation, and whenever `deltaT2_10 ≈ 0` the value of `beta` is essentially
arbitrary (it multiplies a vanishing term). Across all per-hour fits
`corr(|deltaT2_10|, beta) ≈ -0.44`. **The shared-beta fit below fixes this** — it
holds `beta` constant per combo so `deltaT2_10` is stable and well-behaved.

## Shared-beta variant — `src/fit_t2_shared_beta.py`

Same model, but **`beta` is held constant for each `(episode, area)`** while
`T2_0` and `deltaT2_10` still vary hour-by-hour. Fit by **separable / profiled
least squares**: for a fixed `beta` the model is linear in `(T2_0, deltaT2_10)`,
so the inner fit is an ordinary least-squares regression of `T2` on
`x = (release_rate/10)**beta`, done per hour; only the scalar `beta` needs a
nonlinear search (`scipy.optimize.minimize_scalar`, `bounded`) minimizing the
total residual SS across all 121 hours. `beta`'s approximate standard error comes
from the curvature of the profile SSR at the optimum; per-hour `T2_0`/`deltaT2_10`
SEs are OLS SEs conditional on the fitted `beta`.

Fitted `beta` (all sub-linear → saturating dose-response):

| episode | area   | beta  |
|---------|--------|-------|
| 240527  | city   | 0.196 |
| 240527  | region | 0.540 |
| 240727  | city   | 0.175 |
| 240727  | region | 0.275 |

Outputs (`data/output/`):

- `t2_sharedbeta_fits.csv` / `.xlsx` — per `(episode, area, hour)` row with
  `T2_0`, `deltaT2_10` (+ the constant `beta`), `r2`, `rmse`, and **two SE
  flavours per coefficient**: `*_se` (conditional on `beta`) and `*_se_total`
  (includes `beta` uncertainty via the delta method,
  `Var_total = Var(coef|beta) + (∂coef/∂beta)²·Var(beta_hat)`). The `.xlsx` has a
  second `beta` sheet; `t2_sharedbeta_beta.csv` is the same summary.
- `T2_0_sharedbeta.png`, `deltaT2_10_sharedbeta.png` — coefficient vs. hour
  (same 4-series encoding as the per-hour plots), with a 95% CI band drawn from
  `*_se_total`. Because `beta` is pooled over all 121 hours it is tightly
  estimated, so it inflates the `deltaT2_10` band by only ~0.1–5% over the
  conditional SE (most for `240527 region`, which has the largest `beta_se`).
- `beta_sharedbeta.png` — the four constant `beta` values as a bar chart with
  standard-error whiskers (color = episode; city hatched / region solid).

## Outputs (`data/output/`, git-ignored)

- `t2_hourly_fits.csv` / `.xlsx` — one row per `(episode, area, hour)` with
  `T2_0`, `deltaT2_10`, `beta`, their standard errors, `r2`, `rmse`,
  per-dose means (`mean_r0/1/10/100`), `n_points`, `converged`.
- `T2_0.png`, `deltaT2_10.png`, `beta.png` — each coefficient vs. hour, one line
  per `(episode, area)`. **Encoding: line style = area** (dashed city / solid
  region); **color = episode** (blue `240527` / orange `240727`). Shaded band is
  the 95% CI (`±1.96·se`); here `se` comes from the `curve_fit` covariance, which
  already includes `beta`'s uncertainty since all three parameters are fit jointly.

## Model-free sanity check — `src/ctl_anomaly.py`

Works straight from the ensemble members (no regression) to validate the fits:

1. **Control mean/SE** per `(episode, area, hour)` from the 3 ctl members.
2. **Release-minus-control anomalies** per `(episode, area, release_case, hour)`:
   `dT2 = mean(case) - mean(ctl)`, with uncertainty propagated two ways —
   `se_paired = std(case_i - ctl_i, ddof=1)/sqrt(n)` (members are matched across
   scenarios: same meteorology, differing only in release) and
   `se_unpaired = sqrt(se_case^2 + se_ctl^2)`. The paired SE is the proper one;
   empirically it is ~0.47x the unpaired SE (members are strongly correlated
   across scenarios), so pairing roughly halves the uncertainty.

Checks that pass: the **control mean equals the fitted `T2_0`** (identical plots),
and the **r10 anomaly tracks the shared-beta `deltaT2_10`** (RMSD ~0.7 K, residual
dominated by the noisy city signal, which the 4-dose/one-beta fit does not force
through the raw r10 mean).

Outputs (`data/output/`):

- `ctl_mean_se.csv` — control `ctl_mean`, `ctl_se`, `n_ctl` per `(episode, area, hour)`.
- `t2_anomalies.csv` / `.xlsx` — long anomaly table (`dT2`, `se_paired`,
  `se_unpaired`, plus `mean_case/se_case/mean_ctl/se_ctl`); xlsx has an `anomalies`
  and a `ctl` sheet.
- `ctl_mean_se.png` — control T2 vs. hour, 95% band (same 4-series encoding).
- `anomaly_r1.png`, `anomaly_r10.png`, `anomaly_r100.png` — anomaly vs. hour per
  release rate, 95% band from the paired SE.

## Overlay cross-check — `src/overlay_r10.py`

Reads the outputs of `fit_t2_shared_beta.py` and `ctl_anomaly.py` and, in a 2x2
grid (one panel per `(episode, area)`), overlays the fitted shared-beta
`deltaT2_10` (blue, band `deltaT2_10_se_total`) on the model-free r10-ctl anomaly
(orange dashed, band `se_paired`). Each panel title carries the per-combo RMSD.

Result: `240727` agrees almost exactly (RMSD 0.16-0.22 K); `240527` shows a
systematic offset (RMSD 0.71/1.19 K) where the raw r10 anomaly is more negative
than the fitted `deltaT2_10` — the larger-`beta` episode's r10 point sits below
the single power-law curve that must also honor r1 and r100, i.e. a mild sign the
power-law shape fits `240727` better than `240527`.

Outputs (`data/output/`): `deltaT2_10_vs_anomaly_r10.png`,
`r10_vs_deltaT2_10.csv` (per-hour `anomaly_r10`, `deltaT2_10`, both SEs, `diff`).
Run after the shared-beta and anomaly scripts.

## Running

```bash
source .venv/bin/activate        # optional; scripts also run via .venv/bin/python
python src/fit_t2.py
```

Regenerates every file in `data/output/`.

## Layout

```
data/input/T2_summary.csv   source data (read-only)
data/output/                generated CSV/XLSX/PNG (git-ignored)
src/fit_t2_shared_beta_anomaly.py  RECOMMENDED: paired-anomaly shared-beta fit
src/fit_t2_saturation_anomaly.py   alt shape: exponential saturation (release_scale)
src/fit_t2.py               per-hour fit: load → fit → write CSV/XLSX → plot
src/fit_t2_shared_beta.py   shared-beta variant, raw fit (profiled least squares)
src/ctl_anomaly.py          model-free control means + release-minus-ctl anomalies
src/overlay_r10.py          overlay r10 anomaly on shared-beta deltaT2_10
src/scatter_pred_obs.py     predicted-vs-observed anomaly, parametric in hour
.venv/                      numpy pandas scipy statsmodels scikit-learn matplotlib openpyxl
```

## Conventions

- All generated files go under `data/output/`.
- Use the `.venv` interpreter (`.venv/bin/python`). Packages are installed there,
  not in the system Python.
- The source column `evaluatino_area` is misspelled upstream; code renames it to
  `area` on load — do not "fix" the CSV.
