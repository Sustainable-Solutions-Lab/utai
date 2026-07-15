"""Daily-max reduction with a saturating day response as well as a saturating dose.

Extends fit_tmax_saturation.py by letting the reduction build up over the episode
instead of being fit independently per day:

    Tmax_red = T_max_scale_10
             * (1 - exp(-(day - 0.5)/day_scale))
             * (1 - exp(-release_rate/release_scale))
             / (1 - exp(-10/release_scale))

Three free parameters per (episode, area) -- 4 regressions total:
  - T_max_scale_10 (K): the reduction at 10 kt/h once the day response has fully
    built up (day -> infinity).  The reduction on a given day is smaller by the
    day factor; Tmax_red10_day5 below reports the value on the last day.
  - day_scale (days): e-folding time of the buildup.  day_scale -> 0 means the
    full reduction is present on day 1; day_scale >> 5 makes the reduction grow
    nearly linearly across the episode.
  - release_scale (kt/h): dose curvature, as in fit_tmax_saturation.py.

Day blocking, pairing, and the cooling-positive sign convention (ctl - exp) are
inherited unchanged from fit_tmax_saturation.daily_max_reductions.

Fitting: for fixed (day_scale, release_scale) the model is linear in
T_max_scale_10, so the amplitude is closed-form through the origin and only the
two scales are searched (log space, multi-start Nelder-Mead).  Uncertainty uses
the ensemble member as the unit of replication: member-replicate SE for the
amplitude conditional on both scales, plus a delete-one-member jackknife for the
scales and the amplitude's total SE.

Identifiability note: with only days 1..5, day_scale >> 5 makes the day factor
approach (day - 0.5)/day_scale, which trades off against T_max_scale_10 -- only
their ratio is then constrained.  The fit reports day_scale_at_bound so a fit
sitting on that ridge is visible rather than silent; Tmax_red10_day5 stays well
determined either way.

Outputs (data/output/):
    tmax_day_saturation_fit.csv / .xlsx   one row per (episode, area)
    tmax_day_saturation_amplitude.png     T_max_scale_10 bars
    tmax_day_saturation_day_scale.png     day_scale bars
    tmax_day_saturation_release_scale.png release_scale bars
    tmax_day_saturation_curves.png        Tmax_red vs day, points + model curves
    tmax_day_saturation_pred_vs_obs.png   predicted vs observed reduction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize

import fit_t2
from fit_t2 import (
    load_data, OUTPUT_DIR, EPISODE_PALETTE,
    SURFACE, INK, INK_MUTED, GRID, AXIS,
)
from fit_t2_saturation_anomaly import dose_term
from fit_t2_shared_beta_anomaly import sem
from fit_t2_shared_beta import plot_beta_bars
from fit_tmax_saturation import daily_max_reductions

# search ranges in log space: (day_scale in days, release_scale in kt/h)
DAY_LOG_BOUNDS = (np.log(0.02), np.log(1.0e3))
RATE_LOG_BOUNDS = (np.log(0.05), np.log(2.0e4))

# starting points for the multi-start search, in natural units
DAY_STARTS = [0.3, 1.0, 3.0, 30.0]
RATE_STARTS = [1.0, 10.0, 100.0, 1000.0]


def day_term(day: np.ndarray, day_scale: float) -> np.ndarray:
    """1 - exp(-(day - 0.5)/day_scale); ~0 half a day before day 1, 1 as day grows."""
    day = np.asarray(day, dtype=float)
    return -np.expm1(-(day - 0.5) / day_scale)


def _amp_ssr(day, rate, y, day_scale, rate_scale):
    """Through-origin amplitude and residual SS for a candidate pair of scales."""
    x = day_term(day, day_scale) * dose_term(rate, rate_scale)
    denom = float(x @ x)
    a = float(x @ y) / denom if denom > 0 else 0.0
    r = y - a * x
    return a, float(r @ r)


def _profile(day, rate, y):
    """Fit amplitude, day_scale, release_scale to one group's points."""
    def obj(u):
        return _amp_ssr(day, rate, y, np.exp(u[0]), np.exp(u[1]))[1]

    bounds = np.array([DAY_LOG_BOUNDS, RATE_LOG_BOUNDS])

    def clipped(u):
        # Nelder-Mead is unbounded; clamp into the box so the search cannot run
        # off to a degenerate scale and so the reported optimum is attainable.
        return obj(np.clip(u, bounds[:, 0], bounds[:, 1]))

    best = None
    for d0 in DAY_STARTS:
        for r0 in RATE_STARTS:
            u0 = np.array([np.log(d0), np.log(r0)])
            res = minimize(clipped, u0, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
    u = np.clip(best.x, bounds[:, 0], bounds[:, 1])
    day_scale, rate_scale = float(np.exp(u[0])), float(np.exp(u[1]))
    amp, ssr = _amp_ssr(day, rate, y, day_scale, rate_scale)
    return amp, day_scale, rate_scale, ssr


def fit_group(g: pd.DataFrame) -> dict:
    """Fit one (episode, area) with member-replicate + jackknife uncertainties."""
    day = g["day"].to_numpy(float)
    rate = g["release_rate"].to_numpy(float)
    y = g["Tmax_red"].to_numpy(float)
    ens = g["ens"].to_numpy()
    members = sorted(set(ens))
    n = len(members)

    amp, day_scale, rate_scale, ssr = _profile(day, rate, y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ssr / ss_tot) if ss_tot > 0 else np.nan

    # amplitude SE conditional on both scales: per-member through-origin slope
    us = []
    for e in members:
        m = ens == e
        xe = day_term(day[m], day_scale) * dose_term(rate[m], rate_scale)
        denom = float(xe @ xe)
        us.append(float(xe @ y[m]) / denom if denom > 0 else np.nan)
    amp_se_cond = sem(np.asarray(us))

    # delete-one-member jackknife -> scale SEs and amplitude total SE
    jk = {"amp": [], "day": [], "rate": [], "day5": []}
    for e in members:
        m = ens != e
        a_e, d_e, r_e, _ = _profile(day[m], rate[m], y[m])
        jk["amp"].append(a_e)
        jk["day"].append(d_e)
        jk["rate"].append(r_e)
        jk["day5"].append(a_e * float(day_term(5.0, d_e)))

    def jk_se(vals):
        v = np.asarray(vals, dtype=float)
        return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))

    day5 = amp * float(day_term(5.0, day_scale))
    # pegged at either end: upper = linear-in-day ridge, lower = no buildup at all.
    # A pegged scale also gives a degenerate (often exactly 0) jackknife SE.
    at_bound = ""
    if np.isclose(np.log(day_scale), DAY_LOG_BOUNDS[0], atol=1e-3):
        at_bound = "lower"
    elif np.isclose(np.log(day_scale), DAY_LOG_BOUNDS[1], atol=1e-3):
        at_bound = "upper"

    return {
        "Tmax_scale10": amp, "Tmax_scale10_se_cond": amp_se_cond,
        "Tmax_scale10_se": jk_se(jk["amp"]),
        "day_scale": day_scale, "day_scale_se": jk_se(jk["day"]),
        "day_scale_at_bound": at_bound,
        "release_scale": rate_scale, "release_scale_se": jk_se(jk["rate"]),
        "Tmax_red10_day5": day5, "Tmax_red10_day5_se": jk_se(jk["day5"]),
        "r2": r2, "n_points": len(y), "n_members": n,
    }


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def _style_panel(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def _panels(reductions):
    episodes = sorted(reductions["episode"].unique())
    areas = sorted(reductions["area"].unique())
    return [(ep, ar) for ep in episodes for ar in areas]


def plot_curves(reductions: pd.DataFrame, fits: pd.DataFrame, path):
    """Per-member reductions vs day with the fitted curve, one color per rate."""
    fp = fits.set_index(["episode", "area"])
    rates = sorted(reductions["release_rate"].unique())
    rate_color = {r: EPISODE_PALETTE[i % len(EPISODE_PALETTE)]
                  for i, r in enumerate(rates)}
    dgrid = np.linspace(1.0, 5.0, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ep, ar) in zip(np.atleast_2d(axes).ravel(), _panels(reductions)):
        row = fp.loc[(ep, ar)]
        g = reductions[(reductions["episode"] == ep) & (reductions["area"] == ar)]
        for r in rates:
            sub = g[g["release_rate"] == r]
            ax.scatter(sub["day"], sub["Tmax_red"], s=18, alpha=0.55,
                       color=rate_color[r], edgecolor="none")
            curve = (float(row["Tmax_scale10"])
                     * day_term(dgrid, float(row["day_scale"]))
                     * float(dose_term(np.array([r]), float(row["release_scale"]))[0]))
            ax.plot(dgrid, curve, color=rate_color[r], linewidth=2,
                    label=f"{r:g} kt/h")
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        _style_panel(ax, f"{ar} · {ep}", "day", "Tmax_red (K)")
    np.atleast_2d(axes).ravel()[0].legend(frameon=False, fontsize=8,
                                          labelcolor=INK, loc="best")
    fig.suptitle("Daily-max reduction vs. day: members (points) and fitted model (lines)",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_pred_vs_obs(reductions: pd.DataFrame, fits: pd.DataFrame, path):
    fp = fits.set_index(["episode", "area"])
    rates = sorted(reductions["release_rate"].unique())
    rate_color = {r: EPISODE_PALETTE[i % len(EPISODE_PALETTE)]
                  for i, r in enumerate(rates)}

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ep, ar) in zip(np.atleast_2d(axes).ravel(), _panels(reductions)):
        row = fp.loc[(ep, ar)]
        g = reductions[(reductions["episode"] == ep) & (reductions["area"] == ar)]
        rate_arr = g["release_rate"].to_numpy(float)
        pred = (float(row["Tmax_scale10"])
                * day_term(g["day"].to_numpy(float), float(row["day_scale"]))
                * dose_term(rate_arr, float(row["release_scale"])))
        obs = g["Tmax_red"].to_numpy(float)

        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=0)
        for r in rates:
            m = rate_arr == r
            ax.scatter(obs[m], pred[m], s=22, alpha=0.7, color=rate_color[r],
                       edgecolor="none", label=f"{r:g} kt/h")
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        ax.text(0.04, 0.96, f"RMSE {rmse:.2f} K", transform=ax.transAxes,
                ha="left", va="top", color=INK_MUTED, fontsize=9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        _style_panel(ax, f"{ar} · {ep}", "observed Tmax_red (K)",
                     "predicted Tmax_red (K)")
    np.atleast_2d(axes).ravel()[0].legend(frameon=False, fontsize=8,
                                          labelcolor=INK, loc="lower right")
    fig.suptitle("Daily-max reduction: predicted vs. observed (line = 1:1)",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    for i, ep in enumerate(sorted(df["episode"].unique())):
        fit_t2.EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    reductions = daily_max_reductions(df)

    fits = []
    for (episode, area), g in reductions.groupby(["episode", "area"]):
        fits.append({"episode": episode, "area": area, **fit_group(g)})
    fits = pd.DataFrame(fits).sort_values(["episode", "area"]).reset_index(drop=True)

    fits.to_csv(OUTPUT_DIR / "tmax_day_saturation_fit.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "tmax_day_saturation_fit.xlsx",
                        engine="openpyxl") as xw:
        fits.to_excel(xw, sheet_name="fit", index=False)

    plot_beta_bars(fits, OUTPUT_DIR / "tmax_day_saturation_amplitude.png",
                   value_col="Tmax_scale10", se_col="Tmax_scale10_se",
                   ylabel="T_max_scale_10 (K)",
                   title="Daily-max reduction at 10 kt/h, day-asymptotic")
    plot_beta_bars(fits, OUTPUT_DIR / "tmax_day_saturation_day_scale.png",
                   value_col="day_scale", se_col="day_scale_se",
                   ylabel="day_scale (days)",
                   title="Buildup time scale of the daily-max reduction")
    plot_beta_bars(fits, OUTPUT_DIR / "tmax_day_saturation_release_scale.png",
                   value_col="release_scale", se_col="release_scale_se",
                   ylabel="release_scale (kt/h)",
                   title="Dose curvature of the daily-max reduction")
    plot_curves(reductions, fits, OUTPUT_DIR / "tmax_day_saturation_curves.png")
    plot_pred_vs_obs(reductions, fits,
                     OUTPUT_DIR / "tmax_day_saturation_pred_vs_obs.png")

    cols = ["episode", "area", "Tmax_scale10", "Tmax_scale10_se",
            "day_scale", "day_scale_se", "release_scale", "release_scale_se",
            "Tmax_red10_day5", "r2"]
    print(f"Reductions: {len(reductions)} rows")
    print("\nFit per (episode, area):")
    print(fits[cols].to_string(index=False))
    for _, row in fits[fits["day_scale_at_bound"] != ""].iterrows():
        where = row["day_scale_at_bound"]
        why = ("the day response is flat -- no buildup is resolved, so the fit "
               "reduces\n      to the pure dose saturation and day_scale carries no "
               "information"
               if where == "lower" else
               "the day response is effectively linear over days 1-5, so day_scale\n"
               "      trades off against Tmax_scale10 and only their ratio is "
               "constrained")
        print(f"\nNote: {row['area']} {row['episode']}: day_scale pegged at its "
              f"{where} bound;\n      {why}."
              "\n      Its day_scale SE is degenerate; Tmax_red10_day5 is still "
              "well determined.")
    print("\nWrote:\n  tmax_day_saturation_fit.csv / .xlsx")
    print("  tmax_day_saturation_amplitude.png, tmax_day_saturation_day_scale.png,")
    print("  tmax_day_saturation_release_scale.png, tmax_day_saturation_curves.png,")
    print("  tmax_day_saturation_pred_vs_obs.png")


if __name__ == "__main__":
    main()
