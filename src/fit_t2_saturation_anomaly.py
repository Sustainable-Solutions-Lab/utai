"""Exponential-saturation variant of the paired-anomaly shared-scale fit.

Same paired-anomaly framework and error propagation as
fit_t2_shared_beta_anomaly.py, but a different dose-response shape:

    deltaT2 = T2_scale * (1 - exp(-release_rate/release_scale))
                       / (1 - exp(-10/release_scale))

Properties:
  - at release_rate = 0 the factor is 0 (fit through the origin),
  - at release_rate = 10 the factor is 1, so T2_scale (per hour) is exactly the
    perturbation at 10 kt/h -- the same quantity as deltaT2_10, kept under that
    column name so the two functional forms are directly comparable,
  - release_scale (kt/h, one constant per episode x area) sets the curvature:
    release_scale -> infinity gives a linear dose response, release_scale -> 0
    saturates immediately.  It plays the role that beta plays in the power-law
    form.

For a fixed release_scale the model is linear in T2_scale, so the fit is the same
profiled least squares (1-D search over release_scale wrapping a per-hour
through-origin slope) with the same member-replicate + delete-one-member jackknife
uncertainties.  release_scale is searched in log space for scale invariance.

Outputs (data/output/):
    t2_saturation_fit.csv / .xlsx   per-hour T2_scale(=deltaT2_10) (+SEs), release_scale
    t2_saturation_scale.csv         per-combo release_scale + jackknife SE
    deltaT2_10_saturation.png       T2_scale vs hour, +/- 1 SE band (incl. scale)
    release_scale_saturation.png    per-combo release_scale with jackknife SE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import fit_t2
from fit_t2 import load_data, plot_coefficient, OUTPUT_DIR, EPISODE_PALETTE
from fit_t2_shared_beta import plot_beta_bars
from fit_t2_shared_beta_anomaly import build_anomalies, sem

# release_scale search range (kt/h), in log space
LOG_BOUNDS = (np.log(0.05), np.log(2.0e4))


def dose_term(rr: np.ndarray, scale: float) -> np.ndarray:
    """(1 - exp(-rr/scale)) / (1 - exp(-10/scale)); 0 at rr=0. expm1 for accuracy."""
    rr = np.asarray(rr, dtype=float)
    num = -np.expm1(-rr / scale)          # = 1 - exp(-rr/scale)
    den = -np.expm1(-10.0 / scale)
    return num / den


def _prep(cases: pd.DataFrame):
    out = []
    for hour, g in cases.groupby("hour"):
        out.append((int(hour),
                    g["release_rate"].to_numpy(float),
                    g["d"].to_numpy(float),
                    g["ens"].to_numpy()))
    return out


def _ssr_and_deltas(packed, scale: float):
    ssr = 0.0
    deltas = {}
    for hour, rr, d, _ in packed:
        x = dose_term(rr, scale)
        denom = float(x @ x)
        delta = float(x @ d) / denom if denom > 0 else 0.0
        deltas[hour] = delta
        r = d - delta * x
        ssr += float(r @ r)
    return ssr, deltas


def profile_scale(cases: pd.DataFrame):
    packed = _prep(cases)
    res = minimize_scalar(lambda u: _ssr_and_deltas(packed, np.exp(u))[0],
                          bounds=LOG_BOUNDS, method="bounded",
                          options={"xatol": 1e-4})
    scale = float(np.exp(res.x))
    _, deltas = _ssr_and_deltas(packed, scale)
    return scale, deltas, packed


def fit_combo(cases: pd.DataFrame):
    members = sorted(cases["ens"].unique())
    n = len(members)

    scale, deltas, packed = profile_scale(cases)

    # delete-one-member jackknife (whole member removed, pairing preserved)
    jk_scale = []
    jk_delta = {h: [] for h in deltas}
    for e in members:
        sub = cases[cases["ens"] != e]
        s_e, d_e, _ = profile_scale(sub)
        jk_scale.append(s_e)
        for h, val in d_e.items():
            jk_delta[h].append(val)
    jk_scale = np.asarray(jk_scale)
    scale_se = float(np.sqrt((n - 1) / n * np.sum((jk_scale - jk_scale.mean()) ** 2)))

    rows = []
    for hour, rr, d, ens in packed:
        x = dose_term(rr, scale)
        us = []
        for e in members:
            m = ens == e
            xe, de = x[m], d[m]
            denom = float(xe @ xe)
            us.append(float(xe @ de) / denom if denom > 0 else np.nan)
        us = np.asarray(us)
        delta = float(np.nanmean(us))
        se_cond = sem(us)
        de_jk = np.asarray(jk_delta[hour])
        se_total = float(np.sqrt((n - 1) / n * np.sum((de_jk - de_jk.mean()) ** 2)))

        yhat = delta * x
        ss_res = float(np.sum((d - yhat) ** 2))
        ss_tot = float(np.sum((d - d.mean()) ** 2))
        rows.append({
            "hour": hour,
            "deltaT2_10": delta, "release_scale": scale,
            "deltaT2_10_se_cond": se_cond,
            "deltaT2_10_se_total": se_total,
            "r2_anom": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "n_members": n,
        })
    per_hour = pd.DataFrame(rows)
    summary = {"release_scale": scale, "release_scale_se": scale_se, "n_members": n,
               "median_r2_anom": float(per_hour["r2_anom"].median())}
    return per_hour, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    for i, ep in enumerate(sorted(df["episode"].unique())):
        fit_t2.EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    cases = build_anomalies(df)
    per_hour_all, summaries = [], []
    for (episode, area), sub in cases.groupby(["episode", "area"]):
        ph, summ = fit_combo(sub)
        ph.insert(0, "area", area)
        ph.insert(0, "episode", episode)
        per_hour_all.append(ph)
        summaries.append({"episode": episode, "area": area, **summ})

    fits = pd.concat(per_hour_all, ignore_index=True).sort_values(
        ["episode", "area", "hour"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries).sort_values(["episode", "area"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "t2_saturation_fit.csv"
    xlsx_path = OUTPUT_DIR / "t2_saturation_fit.xlsx"
    fits.to_csv(csv_path, index=False)
    summary.to_csv(OUTPUT_DIR / "t2_saturation_scale.csv", index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        fits.to_excel(xw, sheet_name="fits", index=False)
        summary.to_excel(xw, sheet_name="release_scale", index=False)

    plot_coefficient(fits, "deltaT2_10", "T2_scale = deltaT2_10 (K)",
                     "T2_scale (exp-saturation, shared release_scale) vs. hour  "
                     "(band: +/- 1 SE, incl. scale)",
                     OUTPUT_DIR / "deltaT2_10_saturation.png",
                     se_column="deltaT2_10_se_total", z=1.0)
    plot_beta_bars(summary, OUTPUT_DIR / "release_scale_saturation.png",
                   value_col="release_scale", se_col="release_scale_se",
                   ylabel="release_scale (kt/h)",
                   title="Shared release_scale per (episode, area)")

    print("Exp-saturation paired-anomaly fit per (episode, area):")
    print(summary[["episode", "area", "release_scale", "release_scale_se",
                   "median_r2_anom"]].to_string(index=False))
    print(f"\nWrote:\n  {csv_path}\n  {xlsx_path}\n  t2_saturation_scale.csv")
    print("  deltaT2_10_saturation.png, release_scale_saturation.png")


if __name__ == "__main__":
    main()
