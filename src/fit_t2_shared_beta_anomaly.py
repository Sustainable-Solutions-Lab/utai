"""Paired-anomaly shared-beta fit (recommended method).

Instead of fitting raw T2 with a free intercept, this fits the control-subtracted
anomaly, which removes the shared meteorology carried by each matched ensemble
member (e1/e2/e3 are the same weather realization across scenarios).

Per (episode, area):

  baseline   T2_0(h)      = mean over ctl members of T2(h)
             SE[T2_0(h)]  = sd_ctl(h) / sqrt(n_ctl)           (ensemble SE)

  anomaly    d_{s,e}(h)   = T2_{s,e}(h) - T2_{ctl,e}(h)       (paired, per member)
             model        d = deltaT2_10(h) * (release_rate/10) ** beta
             fit through the origin (there is no anomaly at release_rate = 0),
             with beta held CONSTANT per (episode, area) and deltaT2_10 free
             per hour.  Solved by profiled least squares: a 1-D search over beta
             wrapping the closed-form through-origin slope for each hour.

Error propagation treats the ENSEMBLE MEMBER as the unit of replication:

  Each member e supplies a complete paired data set, hence an independent
  through-origin slope u_e(h) = sum_s x_s d_{s,e} / sum_s x_s^2 with
  x_s = (r_s/10)**beta.  Then
       deltaT2_10(h)      = mean_e u_e(h)
       SE(deltaT2_10 | beta) = sd(u_e, ddof=1) / sqrt(n)      (exact, distribution-free;
                                the shared-ctl correlation is inside each u_e)
  beta and the beta-inclusive SE of deltaT2_10 come from a delete-one-member
  jackknife: refit with each whole member removed (its ctl AND experiments drop
  together, preserving the pairing), giving beta_(-e) and deltaT2_10_(-e)(h);
       SE_jk = sqrt((n-1)/n * sum_e (theta_(-e) - mean theta_(-.))^2).
  All standard errors scale ~ 1/sqrt(n_members), so they tighten as members are
  added.  Shaded plot bands are +/- 1 SE (a 95% CI multiplies by t_{n-1}, which
  with n=3 members is ~4.3 -- see CLAUDE.md).

Outputs (data/output/):
    t2_anomaly_fit.csv / .xlsx    per-hour T2_0, deltaT2_10 (+SEs) and a beta sheet
    t2_anomaly_beta.csv           per-combo beta + jackknife SE
    ctl_baseline.png              control T2 mean +/- 1 SE vs hour
    deltaT2_10_anomaly.png        deltaT2_10 vs hour, +/- 1 SE band (incl. beta)
    beta_anomaly.png              per-combo beta with jackknife SE whiskers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import fit_t2
from fit_t2 import load_data, plot_coefficient, OUTPUT_DIR, EPISODE_PALETTE
from fit_t2_shared_beta import plot_beta_bars

BETA_BOUNDS = (0.01, 10.0)


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def build_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's matching ctl member and form the paired anomaly d."""
    ctl = (df[df["scenario"] == "ctl"]
           [["episode", "area", "hour", "ens", "T2"]]
           .rename(columns={"T2": "T2_ctl"}))
    cases = df[df["scenario"] != "ctl"].merge(
        ctl, on=["episode", "area", "hour", "ens"], how="left")
    cases["d"] = cases["T2"] - cases["T2_ctl"]
    return cases


def _prep(cases: pd.DataFrame):
    """Pack one combo into a list of (hour, release_rate, d, ens) arrays."""
    out = []
    for hour, g in cases.groupby("hour"):
        out.append((int(hour),
                    g["release_rate"].to_numpy(float),
                    g["d"].to_numpy(float),
                    g["ens"].to_numpy()))
    return out


def _ssr_and_deltas(packed, beta: float):
    """Total residual SS and per-hour through-origin slope for a candidate beta."""
    ssr = 0.0
    deltas = {}
    for hour, rr, d, _ in packed:
        x = (rr / 10.0) ** beta          # all release rows have rr > 0
        denom = float(x @ x)
        delta = float(x @ d) / denom if denom > 0 else 0.0
        deltas[hour] = delta
        r = d - delta * x
        ssr += float(r @ r)
    return ssr, deltas


def profile_beta(cases: pd.DataFrame):
    """Fit shared beta (+ per-hour deltas) to one combo's anomalies."""
    packed = _prep(cases)
    res = minimize_scalar(lambda b: _ssr_and_deltas(packed, b)[0],
                          bounds=BETA_BOUNDS, method="bounded",
                          options={"xatol": 1e-4})
    beta = float(res.x)
    _, deltas = _ssr_and_deltas(packed, beta)
    return beta, deltas, packed


def fit_combo(cases: pd.DataFrame):
    """Full fit for one (episode, area) with member-replicate uncertainties."""
    members = sorted(cases["ens"].unique())
    n = len(members)

    beta, deltas, packed = profile_beta(cases)

    # delete-one-member jackknife: whole member (ctl + experiments) removed
    jk_beta = []
    jk_delta = {h: [] for h in deltas}
    for e in members:
        sub = cases[cases["ens"] != e]
        b_e, d_e, _ = profile_beta(sub)
        jk_beta.append(b_e)
        for h, val in d_e.items():
            jk_delta[h].append(val)
    jk_beta = np.asarray(jk_beta)
    beta_se = float(np.sqrt((n - 1) / n * np.sum((jk_beta - jk_beta.mean()) ** 2)))

    rows = []
    for hour, rr, d, ens in packed:
        x = (rr / 10.0) ** beta
        # per-member through-origin slope u_e (conditional-on-beta SE)
        us = []
        for e in members:
            m = ens == e
            xe, de = x[m], d[m]
            denom = float(xe @ xe)
            us.append(float(xe @ de) / denom if denom > 0 else np.nan)
        us = np.asarray(us)
        delta = float(np.nanmean(us))       # == pooled through-origin slope
        se_cond = sem(us)

        # jackknife SE of delta (includes beta uncertainty)
        de_jk = np.asarray(jk_delta[hour])
        se_total = float(np.sqrt((n - 1) / n * np.sum((de_jk - de_jk.mean()) ** 2)))

        # anomaly goodness of fit for this hour
        yhat = delta * x
        ss_res = float(np.sum((d - yhat) ** 2))
        ss_tot = float(np.sum((d - d.mean()) ** 2))
        rows.append({
            "hour": hour,
            "deltaT2_10": delta, "beta": beta,
            "deltaT2_10_se_cond": se_cond,     # conditional on beta
            "deltaT2_10_se_total": se_total,   # includes beta (jackknife)
            "r2_anom": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "n_members": n,
        })
    per_hour = pd.DataFrame(rows)
    summary = {"beta": beta, "beta_se": beta_se, "n_members": n,
               "median_r2_anom": float(per_hour["r2_anom"].median())}
    return per_hour, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    for i, ep in enumerate(sorted(df["episode"].unique())):
        fit_t2.EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    # baseline T2_0 = ctl mean +/- ensemble SE
    ctl = df[df["scenario"] == "ctl"]
    baseline = (ctl.groupby(["episode", "area", "hour"])["T2"]
                .agg(T2_0="mean", T2_0_se=sem, n_ctl="count").reset_index())

    cases = build_anomalies(df)
    per_hour_all, summaries = [], []
    for (episode, area), sub in cases.groupby(["episode", "area"]):
        ph, summ = fit_combo(sub)
        ph.insert(0, "area", area)
        ph.insert(0, "episode", episode)
        per_hour_all.append(ph)
        summaries.append({"episode": episode, "area": area, **summ})

    fits = pd.concat(per_hour_all, ignore_index=True)
    fits = fits.merge(baseline, on=["episode", "area", "hour"], how="left")
    fits = fits.sort_values(["episode", "area", "hour"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries).sort_values(["episode", "area"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "t2_anomaly_fit.csv"
    xlsx_path = OUTPUT_DIR / "t2_anomaly_fit.xlsx"
    cols = ["episode", "area", "hour", "n_members", "n_ctl",
            "T2_0", "T2_0_se", "deltaT2_10",
            "deltaT2_10_se_cond", "deltaT2_10_se_total", "beta", "r2_anom"]
    fits = fits[cols]
    fits.to_csv(csv_path, index=False)
    summary.to_csv(OUTPUT_DIR / "t2_anomaly_beta.csv", index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        fits.to_excel(xw, sheet_name="fits", index=False)
        summary.to_excel(xw, sheet_name="beta", index=False)

    # plots (bands are +/- 1 SE; z=1)
    plot_coefficient(baseline, "T2_0", "ctl T2 (K)",
                     "Baseline T2_0 = control mean +/- 1 SE vs. hour",
                     OUTPUT_DIR / "ctl_baseline.png", se_column="T2_0_se", z=1.0)
    plot_coefficient(fits, "deltaT2_10", "deltaT2_10 (K)",
                     "deltaT2_10 (paired anomaly, shared beta) vs. hour  "
                     "(band: +/- 1 SE, incl. beta)",
                     OUTPUT_DIR / "deltaT2_10_anomaly.png",
                     se_column="deltaT2_10_se_total", z=1.0)
    plot_beta_bars(summary, OUTPUT_DIR / "beta_anomaly.png")

    print("Paired-anomaly shared-beta fit per (episode, area):")
    print(summary[["episode", "area", "beta", "beta_se", "median_r2_anom"]]
          .to_string(index=False))
    print(f"\nWrote:\n  {csv_path}\n  {xlsx_path}\n  t2_anomaly_beta.csv")
    print("  ctl_baseline.png, deltaT2_10_anomaly.png, beta_anomaly.png")


if __name__ == "__main__":
    main()
