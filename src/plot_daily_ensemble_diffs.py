"""Difference (experiment - control) ensemble whisker plots.

The companion to `plot_daily_ensemble_stats.py`.  For each ensemble member we
take `experiment - control` on each daily temperature quantity, aligned member
to member (e1-e1, e2-e2, e3-e3) within the same episode/area/day, then summarize
the 3-member spread of that difference.  Cooling of the release runs is negative.

Because release rate is no longer a reference axis, all three rates (1, 10, 100
kt/h) share a panel, offset and colored by rate.

Intermediate product (to data/output/):
    daily_ensemble_diffs.csv / .xlsx   one row per (episode, area, rate, day),
                                       12 stat columns (3 quantities x 4 stats).

Plots (to data/output/), one page per quantity, 2 rows (episode) x 2 cols (area):
    daily_ensemble_mean_temp_diff.png   Δ daily-mean temperature
    daily_ensemble_max_temp_diff.png    Δ daily-max temperature
    daily_ensemble_min_temp_diff.png    Δ daily-min temperature

Each panel overlays rates 1/10/100 (offset, colored by rate) about a zero line.
Per day and rate: a dot at the ensemble median, an x at the ensemble mean, and
whiskers from the ensemble min to the ensemble max of the member differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from fit_t2 import OUTPUT_DIR, SURFACE, INK, AXIS
from plot_daily_ensemble_stats import (
    member_daily, ylim_for, _style, EPISODES, AREAS,
)


def _draw_diff_series(ax, sub, dq, x, color):
    """Median dots joined by a line, with min->max whiskers (no mean marker)."""
    sub = sub.sort_values("day")
    day = sub["day"].to_numpy(float)
    median = sub[f"{dq}_ensmedian"].to_numpy(float)
    lo = sub[f"{dq}_ensmin"].to_numpy(float)
    hi = sub[f"{dq}_ensmax"].to_numpy(float)
    ax.errorbar(day + x, median, yerr=[median - lo, hi - median],
                fmt="-o", markersize=5, color=color, ecolor=color,
                elinewidth=1.4, linewidth=1.5, capsize=3, zorder=3)

# base daily quantities and the difference-column / label used per page
QUANTITIES = [
    ("Tmean", "dTmean", "Δ daily-mean temperature"),
    ("Tmax", "dTmax", "Δ daily-max temperature"),
    ("Tmin", "dTmin", "Δ daily-min temperature"),
]

RATES = [1.0, 10.0, 100.0]
# color = release rate (same hues as plot_tmax_raw.RATE_COLOR)
RATE_COLOR = {1.0: "#2a78d6", 10.0: "#eb6834", 100.0: "#008300"}

OUTPUTS = {
    "dTmean": "daily_ensemble_mean_temp_diff.png",
    "dTmax": "daily_ensemble_max_temp_diff.png",
    "dTmin": "daily_ensemble_min_temp_diff.png",
}


def build_diffs() -> pd.DataFrame:
    """Ensemble stats of the per-member experiment-minus-control differences."""
    mem = member_daily()
    base = ["Tmean", "Tmax", "Tmin"]

    ctl = (mem[mem["release_rate"] == 0.0]
           .drop(columns="release_rate")
           .rename(columns={q: f"{q}_c" for q in base}))
    exp = mem[mem["release_rate"] != 0.0].merge(
        ctl, on=["episode", "area", "ens", "day"], how="left")
    for q in base:
        exp[f"d{q}"] = exp[q] - exp[f"{q}_c"]  # experiment - control

    gkeys = ["episode", "area", "release_rate", "day"]
    agg = {}
    for q in base:
        dq = f"d{q}"
        agg[f"{dq}_ensmean"] = (dq, "mean")
        agg[f"{dq}_ensmedian"] = (dq, "median")
        agg[f"{dq}_ensmin"] = (dq, "min")
        agg[f"{dq}_ensmax"] = (dq, "max")
    diffs = exp.groupby(gkeys, as_index=False).agg(**agg)
    return diffs.sort_values(gkeys).reset_index(drop=True)


def plot_quantity(diffs: pd.DataFrame, dq: str, label: str, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    # one shared y-range for the whole page (nearest 1 °C, always spanning zero)
    ylim = ylim_for(diffs, dq, step=1, include_zero=True)
    for i, episode in enumerate(EPISODES):
        for j, area in enumerate(AREAS):
            ax = axes[i, j]
            ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
            base = diffs[(diffs["episode"] == episode) & (diffs["area"] == area)]
            for rate in RATES:
                sub = base[base["release_rate"] == rate]
                _draw_diff_series(ax, sub, dq, 0.0, RATE_COLOR[rate])
            xlabel = "day" if i == len(EPISODES) - 1 else ""
            _style(ax, f"{episode} · {area}", xlabel, "Δ temperature (°C)")
            ax.set_ylim(*ylim)
            ax.set_xlim(0.5, 5.5)
            ax.xaxis.set_major_locator(MultipleLocator(1))

    handles = [
        Line2D([0], [0], color=RATE_COLOR[r], marker="o", linestyle="-",
               markersize=6, label=f"{r:g} kt/h") for r in RATES
    ] + [
        Line2D([0], [0], color=INK, marker="o", linestyle="-", markersize=6,
               label="ensemble median"),
        Line2D([0], [0], color=INK, marker="|", linestyle="", markersize=12,
               label="whisker: ensemble min–max"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Ensemble {label} (experiment − control) vs. day",
                 color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diffs = build_diffs()
    diffs.to_csv(OUTPUT_DIR / "daily_ensemble_diffs.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "daily_ensemble_diffs.xlsx",
                        engine="openpyxl") as xw:
        diffs.to_excel(xw, sheet_name="daily_ensemble_diffs", index=False)

    for _, dq, label in QUANTITIES:
        plot_quantity(diffs, dq, label, OUTPUT_DIR / OUTPUTS[dq])

    print("Wrote:")
    print("  daily_ensemble_diffs.csv")
    print("  daily_ensemble_diffs.xlsx")
    for name in OUTPUTS.values():
        print(f"  {name}")


if __name__ == "__main__":
    main()
