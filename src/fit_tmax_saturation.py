"""Daily-max temperature reduction and its saturation dose-response.

For each ensemble member we take the daily maximum T2 in 24-hour blocks, subtract
the matched control's daily maximum, and fit the exponential-saturation
dose-response to the resulting reduction -- once with all days grouped and once
per day.

Day blocking: hour 1 is dropped; hours 2..121 form 5 clean 24-hour blocks,
day = (hour - 2)//24 + 1.

Pairing (per member): Tmax_red_{r,e,d} = max_{h in d} T2_{ctl,e}(h)
                                       - max_{h in d} T2_{r,e}(h).
Sign convention: control minus experiment, so cooling of the daily high is a
POSITIVE reduction. (The hourly anomaly fits use the opposite sign, exp - ctl.)

Model (through the origin over r in {1,10,100}):
    Tmax_red = A * (1 - exp(-r/release_scale)) / (1 - exp(-10/release_scale))
with A = Tmax_red10 (the reduction at 10 kt/h) free per fit and release_scale
constant per fit. Solved by profiled least squares (closed-form through-origin
amplitude inside a 1-D log-space search over release_scale). Uncertainty uses the
ensemble member as the unit of replication: member-replicate SE for A conditional
on release_scale, and a delete-one-member jackknife for release_scale and A's
total SE.

Outputs (data/output/):
    tmax_reductions.csv              processed per-member daily-max reductions
    tmax_saturation_grouped.csv      one fit per (episode, area), all days pooled
    tmax_saturation_byday.csv        one fit per (episode, area, day)
    tmax_saturation_fit.xlsx         sheets: grouped, byday, reductions
    tmax_amplitude_grouped.png       grouped Tmax_red10 bar chart
    tmax_release_scale_grouped.png   grouped release_scale bar chart
    tmax_amplitude_byday.png         Tmax_red10 vs day, 4 series, +/-1 SE
    tmax_release_scale_byday.png     release_scale vs day, 4 series, +/-1 SE
    tmax_pred_vs_obs.png             predicted vs observed reduction (grouped)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize_scalar

import fit_t2
from fit_t2 import (
    load_data, OUTPUT_DIR, EPISODE_PALETTE,
    SURFACE, INK, INK_MUTED, GRID, AXIS, AREA_STYLE,
)
from fit_t2_saturation_anomaly import dose_term, LOG_BOUNDS
from fit_t2_shared_beta_anomaly import sem
from fit_t2_shared_beta import plot_beta_bars

RELEASE_RATES = [1.0, 10.0, 100.0]  # doses used in the fit (control excluded)


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------

def daily_max_reductions(df: pd.DataFrame) -> pd.DataFrame:
    """Per-member daily-max reduction Tmax_exp - Tmax_ctl."""
    d = df[df["hour"] != 1].copy()
    d["day"] = (d["hour"] - 2) // 24 + 1

    keys = ["episode", "area", "scenario", "release_rate", "ens", "day"]
    dmax = d.groupby(keys, as_index=False)["T2"].max().rename(columns={"T2": "T_max"})

    ctl = (dmax[dmax["scenario"] == "ctl"]
           [["episode", "area", "ens", "day", "T_max"]]
           .rename(columns={"T_max": "T_max_ctl"}))
    cases = dmax[dmax["scenario"] != "ctl"].merge(
        ctl, on=["episode", "area", "ens", "day"], how="left")
    cases = cases.rename(columns={"T_max": "T_max_exp"})
    # control minus experiment: cooling of the daily high is a positive reduction
    cases["Tmax_red"] = cases["T_max_ctl"] - cases["T_max_exp"]
    return cases.sort_values(
        ["episode", "area", "release_rate", "ens", "day"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# saturation fit (single amplitude + single release_scale per group)
# --------------------------------------------------------------------------

def _amp_ssr(rate, y, scale):
    """Through-origin amplitude A and residual SS for a candidate scale."""
    x = dose_term(rate, scale)
    denom = float(x @ x)
    a = float(x @ y) / denom if denom > 0 else 0.0
    r = y - a * x
    return a, float(r @ r)


def _profile(rate, y):
    """Fit A and release_scale to one group's (rate, Tmax_red) points."""
    res = minimize_scalar(lambda u: _amp_ssr(rate, y, np.exp(u))[1],
                          bounds=LOG_BOUNDS, method="bounded",
                          options={"xatol": 1e-4})
    scale = float(np.exp(res.x))
    a, ssr = _amp_ssr(rate, y, scale)
    return a, scale, ssr


def fit_group(g: pd.DataFrame) -> dict:
    """Fit one group with member-replicate + jackknife uncertainties."""
    rate = g["release_rate"].to_numpy(float)
    y = g["Tmax_red"].to_numpy(float)
    ens = g["ens"].to_numpy()
    members = sorted(set(ens))
    n = len(members)

    amp, scale, ssr = _profile(rate, y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ssr / ss_tot) if ss_tot > 0 else np.nan

    # conditional-on-scale amplitude SE: per-member through-origin slope
    us = []
    for e in members:
        m = ens == e
        xe = dose_term(rate[m], scale)
        denom = float(xe @ xe)
        us.append(float(xe @ y[m]) / denom if denom > 0 else np.nan)
    amp_se_cond = sem(np.asarray(us))

    # delete-one-member jackknife -> scale SE and amplitude total SE
    jk_amp, jk_scale = [], []
    for e in members:
        m = ens != e
        a_e, s_e, _ = _profile(rate[m], y[m])
        jk_amp.append(a_e)
        jk_scale.append(s_e)
    jk_amp = np.asarray(jk_amp)
    jk_scale = np.asarray(jk_scale)
    amp_se_total = float(np.sqrt((n - 1) / n * np.sum((jk_amp - jk_amp.mean()) ** 2)))
    scale_se = float(np.sqrt((n - 1) / n * np.sum((jk_scale - jk_scale.mean()) ** 2)))

    return {
        "Tmax_red10": amp, "Tmax_red10_se_cond": amp_se_cond,
        "Tmax_red10_se": amp_se_total,
        "release_scale": scale, "release_scale_se": scale_se,
        "r2": r2, "n_points": len(y), "n_members": n,
    }


# --------------------------------------------------------------------------
# plotting (local helpers; day axis, so not the hour-specific fit_t2 helpers)
# --------------------------------------------------------------------------

def _style_day_axes(ax, title, ylabel, xlabel="day"):
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_byday(byday: pd.DataFrame, column: str, se_column: str,
               ylabel: str, title: str, path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for area in sorted(byday["area"].unique()):
        for episode in sorted(byday["episode"].unique()):
            sub = byday[(byday["area"] == area)
                        & (byday["episode"] == episode)].sort_values("day")
            color = fit_t2.EPISODE_COLOR.get(episode, INK)
            se = sub[se_column].to_numpy(float)
            ax.fill_between(sub["day"], sub[column] - se, sub[column] + se,
                            color=color, alpha=0.13, linewidth=0)
            ax.plot(sub["day"], sub[column], color=color,
                    linestyle=AREA_STYLE.get(area, "-"), linewidth=2,
                    marker="o", markersize=4, label=f"{area} · {episode}")
    _style_day_axes(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_pred_vs_obs(reductions: pd.DataFrame, grouped: pd.DataFrame, path):
    """Predicted vs observed Tmax_red per (episode, area), colored by rate."""
    gp = grouped.set_index(["episode", "area"])
    rates = sorted(reductions["release_rate"].unique())
    rate_color = {r: EPISODE_PALETTE[i % len(EPISODE_PALETTE)]
                  for i, r in enumerate(rates)}

    episodes = sorted(reductions["episode"].unique())
    areas = sorted(reductions["area"].unique())
    combos = [(ep, ar) for ep in episodes for ar in areas]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    for ax, (ep, ar) in zip(axes.ravel(), combos):
        row = gp.loc[(ep, ar)]
        g = reductions[(reductions["episode"] == ep) & (reductions["area"] == ar)]
        rate_arr = g["release_rate"].to_numpy(float)
        pred = float(row["Tmax_red10"]) * dose_term(rate_arr, float(row["release_scale"]))
        obs = g["Tmax_red"].to_numpy(float)

        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=0)
        for r in rates:
            m = g["release_rate"].to_numpy(float) == r
            ax.scatter(obs[m], np.asarray(pred)[m], s=22, alpha=0.7,
                       color=rate_color[r], edgecolor="none",
                       label=f"{r:g} kt/h")
        rmse = float(np.sqrt(np.mean((np.asarray(pred) - obs) ** 2)))
        ax.text(0.04, 0.96, f"RMSE {rmse:.2f} K", transform=ax.transAxes,
                ha="left", va="top", color=INK_MUTED, fontsize=9)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.set_title(f"{ar} · {ep}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("observed Tmax_red (K)", color=INK_MUTED, fontsize=9)
        ax.set_ylabel("predicted Tmax_red (K)", color=INK_MUTED, fontsize=9)

    axes.ravel()[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle("Daily-max reduction: predicted vs. observed (grouped fit; line = 1:1)",
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
    red_out = reductions[["episode", "area", "scenario", "release_rate", "ens",
                          "day", "T_max_exp", "T_max_ctl", "Tmax_red"]]
    red_out.to_csv(OUTPUT_DIR / "tmax_reductions.csv", index=False)

    # grouped fit: all days pooled, one per (episode, area)
    grouped = []
    for (episode, area), g in reductions.groupby(["episode", "area"]):
        grouped.append({"episode": episode, "area": area, **fit_group(g)})
    grouped = pd.DataFrame(grouped).sort_values(
        ["episode", "area"]).reset_index(drop=True)

    # per-day fit: one per (episode, area, day)
    byday = []
    for (episode, area, day), g in reductions.groupby(["episode", "area", "day"]):
        byday.append({"episode": episode, "area": area, "day": int(day),
                      **fit_group(g)})
    byday = pd.DataFrame(byday).sort_values(
        ["episode", "area", "day"]).reset_index(drop=True)

    grouped.to_csv(OUTPUT_DIR / "tmax_saturation_grouped.csv", index=False)
    byday.to_csv(OUTPUT_DIR / "tmax_saturation_byday.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "tmax_saturation_fit.xlsx",
                        engine="openpyxl") as xw:
        grouped.to_excel(xw, sheet_name="grouped", index=False)
        byday.to_excel(xw, sheet_name="byday", index=False)
        red_out.to_excel(xw, sheet_name="reductions", index=False)

    # plots
    plot_beta_bars(grouped, OUTPUT_DIR / "tmax_amplitude_grouped.png",
                   value_col="Tmax_red10", se_col="Tmax_red10_se",
                   ylabel="Tmax_red at 10 kt/h (K)",
                   title="Daily-max reduction at 10 kt/h (grouped)")
    plot_beta_bars(grouped, OUTPUT_DIR / "tmax_release_scale_grouped.png",
                   value_col="release_scale", se_col="release_scale_se",
                   ylabel="release_scale (kt/h)",
                   title="Daily-max release_scale (grouped)")
    plot_byday(byday, "Tmax_red10", "Tmax_red10_se",
               "Tmax_red at 10 kt/h (K)",
               "Daily-max reduction at 10 kt/h vs. day  (band: +/- 1 SE)",
               OUTPUT_DIR / "tmax_amplitude_byday.png")
    plot_byday(byday, "release_scale", "release_scale_se",
               "release_scale (kt/h)",
               "Daily-max release_scale vs. day  (band: +/- 1 SE)",
               OUTPUT_DIR / "tmax_release_scale_byday.png")
    plot_pred_vs_obs(reductions, grouped, OUTPUT_DIR / "tmax_pred_vs_obs.png")

    print(f"Reductions: {len(reductions)} rows")
    print("\nGrouped fit per (episode, area):")
    print(grouped[["episode", "area", "Tmax_red10", "Tmax_red10_se",
                   "release_scale", "release_scale_se", "r2"]].to_string(index=False))
    print(f"\nWrote:\n  tmax_reductions.csv\n  tmax_saturation_grouped.csv"
          f"\n  tmax_saturation_byday.csv\n  tmax_saturation_fit.xlsx")
    print("  tmax_amplitude_grouped.png, tmax_release_scale_grouped.png,")
    print("  tmax_amplitude_byday.png, tmax_release_scale_byday.png, tmax_pred_vs_obs.png")


if __name__ == "__main__":
    main()
