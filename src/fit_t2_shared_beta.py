"""Shared-beta variant of the T2 dose-response fit.

Same model as src/fit_t2.py:

    T2 = T2_0 + deltaT2_10 * (release_rate / 10) ** beta

but beta is held CONSTANT for each (episode, area) combination, while T2_0 and
deltaT2_10 still vary hour-by-hour.

Method: separable / profiled least squares.  For a fixed beta the model is
linear in (T2_0, deltaT2_10), so the inner fit for each hour is ordinary least
squares of T2 on x = (release_rate/10)**beta.  Only the scalar beta needs a
nonlinear search (1-D), which we do with scipy.optimize.minimize_scalar over the
total residual sum of squares across all 121 hours.

Outputs (data/output/):
    t2_sharedbeta_fits.csv / .xlsx   per-hour T2_0, deltaT2_10 (+ the combo beta)
                                     plus a 'beta' summary sheet in the xlsx
    T2_0_sharedbeta.png, deltaT2_10_sharedbeta.png, beta_sharedbeta.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import fit_t2  # sibling module (run from src/ or with src on the path)
from fit_t2 import (
    load_data, plot_coefficient, style_axes,
    OUTPUT_DIR, SURFACE, INK, INK_MUTED, EPISODE_COLOR, EPISODE_PALETTE,
)

BETA_BOUNDS = (0.01, 10.0)


def dose_term(rr: np.ndarray, beta: float) -> np.ndarray:
    """(rr/10)**beta, with the rr=0 term pinned to 0."""
    rr = np.asarray(rr, dtype=float)
    ratio = np.where(rr > 0, rr / 10.0, 1.0)
    return np.where(rr > 0, np.power(ratio, beta), 0.0)


def ols(x: np.ndarray, y: np.ndarray):
    """OLS of y on [1, x].  Returns (intercept, slope, residuals, XtX_inv)."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    xtx_inv = np.linalg.inv(X.T @ X)
    return coef[0], coef[1], resid, xtx_inv


def total_ssr(beta: float, hours: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Sum of residual SS over all hours for a candidate beta (the profile)."""
    s = 0.0
    for rr, y in hours:
        _, _, resid, _ = ols(dose_term(rr, beta), y)
        s += float(resid @ resid)
    return s


def fit_combo(sub: pd.DataFrame):
    """Fit one (episode, area): shared beta + per-hour T2_0/deltaT2_10."""
    grouped = [
        (g["release_rate"].to_numpy(float), g["T2"].to_numpy(float))
        for _, g in sub.groupby("hour")
    ]

    res = minimize_scalar(
        total_ssr, args=(grouped,), bounds=BETA_BOUNDS, method="bounded",
        options={"xatol": 1e-4},
    )
    beta = float(res.x)

    # bookkeeping for an approximate beta standard error via profile curvature
    n_obs = int(sub.shape[0])
    n_hours = len(grouped)
    n_params = 2 * n_hours + 1               # T2_0,delta per hour + one beta
    ssr_min = total_ssr(beta, grouped)
    sigma2 = ssr_min / max(n_obs - n_params, 1)
    db = 1e-3
    ssr_p = total_ssr(min(beta + db, BETA_BOUNDS[1]), grouped)
    ssr_m = total_ssr(max(beta - db, BETA_BOUNDS[0]), grouped)
    d2 = (ssr_p - 2 * ssr_min + ssr_m) / (db * db)   # d^2 SSR / d beta^2
    beta_se = float(np.sqrt(2 * sigma2 / d2)) if d2 > 0 else np.nan

    # per-hour coefficients.  Two flavours of standard error:
    #   *_se        conditional on beta (plain OLS, treats beta as known)
    #   *_se_total  includes beta uncertainty via the delta method:
    #               Var_total = Var(coef|beta) + (d coef/d beta)^2 * Var(beta_hat)
    db = 1e-3
    beta_var = beta_se ** 2 if np.isfinite(beta_se) else 0.0
    rows = []
    for (hour, g) in sub.groupby("hour"):
        rr = g["release_rate"].to_numpy(float)
        y = g["T2"].to_numpy(float)
        t2_0, delta, resid, xtx_inv = ols(dose_term(rr, beta), y)
        n = len(y)
        ss_res = float(resid @ resid)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        s2 = ss_res / max(n - 2, 1)
        se = np.sqrt(np.diag(s2 * xtx_inv))  # conditional on beta

        # d coef / d beta by central difference, then propagate Var(beta_hat)
        tp0, dp, _, _ = ols(dose_term(rr, beta + db), y)
        tm0, dm, _, _ = ols(dose_term(rr, beta - db), y)
        dT0_db = (tp0 - tm0) / (2 * db)
        dDelta_db = (dp - dm) / (2 * db)
        se_t0_tot = float(np.sqrt(se[0] ** 2 + (dT0_db ** 2) * beta_var))
        se_d_tot = float(np.sqrt(se[1] ** 2 + (dDelta_db ** 2) * beta_var))

        rows.append({
            "hour": int(hour),
            "T2_0": t2_0, "deltaT2_10": delta, "beta": beta,
            "T2_0_se": se[0], "deltaT2_10_se": se[1],
            "T2_0_se_total": se_t0_tot, "deltaT2_10_se_total": se_d_tot,
            "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "rmse": float(np.sqrt(ss_res / n)),
            "n_points": n,
        })
    per_hour = pd.DataFrame(rows)

    summary = {
        "beta": beta, "beta_se": beta_se,
        "n_hours": n_hours, "n_obs": n_obs,
        "total_ssr": ssr_min,
        "median_r2": float(per_hour["r2"].median()),
        "rmse": float(np.sqrt(ssr_min / n_obs)),
    }
    return per_hour, summary


def plot_beta_bars(summary: pd.DataFrame, path, value_col="beta", se_col="beta_se",
                   ylabel="beta", title="Shared beta per (episode, area)"):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    labels = [f"{a}\n{e}" for a, e in zip(summary["area"], summary["episode"])]
    x = np.arange(len(summary))
    colors = [EPISODE_COLOR.get(e, INK) for e in summary["episode"]]
    # hatch denotes area (city hatched, region solid) to match the line plots
    hatch = ["//" if a == "city" else "" for a in summary["area"]]
    vals = summary[value_col].to_numpy(float)
    ses = summary[se_col].to_numpy(float)
    bars = ax.bar(x, vals, yerr=ses, capsize=4,
                  color=colors, edgecolor=INK, linewidth=0.6)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    off = 0.02 * (np.nanmax(vals) if np.nanmax(vals) > 0 else 1.0)
    for xi, (bv, se) in enumerate(zip(vals, ses)):
        ax.text(xi, bv + (se if np.isfinite(se) else 0) + off,
                f"{bv:.3g}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax, title, ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # color per episode (line style already encodes area in fit_t2.plot_coefficient)
    for i, ep in enumerate(sorted(df["episode"].unique())):
        fit_t2.EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    per_hour_all, summaries = [], []
    for (episode, area), sub in df.groupby(["episode", "area"]):
        ph, summ = fit_combo(sub)
        ph.insert(0, "area", area)
        ph.insert(0, "episode", episode)
        per_hour_all.append(ph)
        summaries.append({"episode": episode, "area": area, **summ})

    fits = pd.concat(per_hour_all, ignore_index=True).sort_values(
        ["episode", "area", "hour"]).reset_index(drop=True)
    summary = pd.DataFrame(summaries).sort_values(["episode", "area"]).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "t2_sharedbeta_fits.csv"
    xlsx_path = OUTPUT_DIR / "t2_sharedbeta_fits.xlsx"
    fits.to_csv(csv_path, index=False)
    summary.to_csv(OUTPUT_DIR / "t2_sharedbeta_beta.csv", index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        fits.to_excel(xw, sheet_name="fits", index=False)
        summary.to_excel(xw, sheet_name="beta", index=False)

    plot_coefficient(fits, "T2_0", "T2_0 (K)",
                     "Control temperature T2_0 vs. hour (shared beta)  "
                     "(band: 95% CI, incl. beta uncertainty)",
                     OUTPUT_DIR / "T2_0_sharedbeta.png", se_column="T2_0_se_total")
    plot_coefficient(fits, "deltaT2_10", "deltaT2_10 (K)",
                     "Perturbation at 10 kt/h (shared beta) vs. hour  "
                     "(band: 95% CI, incl. beta uncertainty)",
                     OUTPUT_DIR / "deltaT2_10_sharedbeta.png",
                     se_column="deltaT2_10_se_total")
    plot_beta_bars(summary, OUTPUT_DIR / "beta_sharedbeta.png")

    print("Shared-beta fit per (episode, area):")
    print(summary[["episode", "area", "beta", "beta_se", "median_r2", "rmse"]]
          .to_string(index=False))
    print(f"\nWrote:\n  {csv_path}\n  {xlsx_path}")
    print("  t2_sharedbeta_beta.csv")
    print("  T2_0_sharedbeta.png, deltaT2_10_sharedbeta.png, beta_sharedbeta.png")


if __name__ == "__main__":
    main()
