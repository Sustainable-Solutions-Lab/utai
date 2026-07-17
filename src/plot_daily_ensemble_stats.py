"""Daily ensemble-statistics table and per-metric whisker plots.

Aggregates the raw T2 data into the repo's standard 5-day blocks (hour 1
dropped; day = (hour - 2)//24 + 1) and, for each (episode, area, release_rate,
day), summarizes the 3-member ensemble spread of three daily temperature
quantities:

    Tmean  -- the daily mean temperature (mean over the day's 24 hours)
    Tmax   -- the daily maximum temperature
    Tmin   -- the daily minimum temperature

For each quantity we record the ensemble mean / median / min / max across the
three members (e1, e2, e3).  All temperatures are in degrees Celsius.

Intermediate product (to data/output/):
    daily_ensemble_stats.csv / .xlsx   one row per (episode, area, rate, day),
                                       12 stat columns (3 quantities x 4 stats),
                                       control (rate 0) included.

Plots (to data/output/), one page per quantity, 6 rows x 2 cols
(rows: 240527 @ 1/10/100 then 240727 @ 1/10/100; cols: city, region):
    daily_ensemble_mean_temp.png   daily-mean temperature
    daily_ensemble_max_temp.png    daily-max temperature
    daily_ensemble_min_temp.png    daily-min temperature

Each panel overlays the experiment (episode-colored) beside its matching control
(neutral gray).  Per day: a dot at the ensemble median, an x at the ensemble
mean, and whiskers from the ensemble min to the ensemble max.  Every panel on
every page shares a fixed y-range so all cases are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from fit_t2 import (
    OUTPUT_DIR, SURFACE, INK, INK_MUTED, GRID, AXIS,
    EPISODE_PALETTE, EPISODE_COLOR, load_data,
)

# The three daily quantities we summarize, and their per-member aggregation.
QUANTITIES = [
    ("Tmean", "mean", "daily-mean temperature"),
    ("Tmax", "max", "daily-max temperature"),
    ("Tmin", "min", "daily-min temperature"),
]
ENS_STATS = ["ensmean", "ensmedian", "ensmin", "ensmax"]

# Rows (episode, rate) x cols (area) layout for the panel grid.
EPISODES = ["240527", "240727"]
RATES = [1.0, 10.0, 100.0]
AREAS = ["city", "region"]
ROW_ORDER = [(ep, r) for ep in EPISODES for r in RATES]

# y-axis is shared within a page and framed to that page's own data, rounded
# outward to the nearest YSTEP degrees C.
YSTEP = 5
CTL_COLOR = INK_MUTED  # control series drawn in neutral gray
DX = 0.15              # horizontal offset separating control from experiment


def ylim_for(stats: pd.DataFrame, q: str, step: float = YSTEP,
             include_zero: bool = False):
    """Round the whisker extremes (ensmin..ensmax) out to the nearest `step`."""
    lo = stats[f"{q}_ensmin"].min()
    hi = stats[f"{q}_ensmax"].max()
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    return (np.floor(lo / step) * step, np.ceil(hi / step) * step)


def member_daily() -> pd.DataFrame:
    """Per-(episode, area, release_rate, ens, day) daily Tmean/Tmax/Tmin in °C."""
    df = load_data()
    df["T2C"] = df["T2"] - 273.15  # Kelvin -> Celsius
    d = df[df["hour"] != 1].copy()
    d["day"] = (d["hour"] - 2) // 24 + 1  # 5 clean 24-hour blocks, days 1..5
    keys = ["episode", "area", "release_rate", "ens", "day"]
    return d.groupby(keys, as_index=False).agg(
        Tmean=("T2C", "mean"), Tmax=("T2C", "max"), Tmin=("T2C", "min"))


def build_stats() -> pd.DataFrame:
    """Per-(episode, area, rate, day) ensemble stats of the daily quantities."""
    member = member_daily()

    # ensemble stats across the 3 members, for each daily quantity
    gkeys = ["episode", "area", "release_rate", "day"]
    agg = {}
    for q, _, _ in QUANTITIES:
        agg[f"{q}_ensmean"] = (q, "mean")
        agg[f"{q}_ensmedian"] = (q, "median")
        agg[f"{q}_ensmin"] = (q, "min")
        agg[f"{q}_ensmax"] = (q, "max")
    stats = member.groupby(gkeys, as_index=False).agg(**agg)
    return stats.sort_values(gkeys).reset_index(drop=True)


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def _draw_series(ax, sub, q, x, color):
    """Median dot + min->max whiskers + ensemble-mean x for one series."""
    sub = sub.sort_values("day")
    day = sub["day"].to_numpy(float)
    median = sub[f"{q}_ensmedian"].to_numpy(float)
    lo = sub[f"{q}_ensmin"].to_numpy(float)
    hi = sub[f"{q}_ensmax"].to_numpy(float)
    mean = sub[f"{q}_ensmean"].to_numpy(float)
    ax.errorbar(day + x, median, yerr=[median - lo, hi - median],
                fmt="o", markersize=5, color=color, ecolor=color,
                elinewidth=1.4, capsize=3, zorder=3)
    ax.scatter(day + x, mean, marker="x", s=28, color=color,
               linewidths=1.4, zorder=4)


def plot_quantity(stats: pd.DataFrame, q: str, label: str, path):
    fig, axes = plt.subplots(6, 2, figsize=(11, 16), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    # one shared y-range per episode (the three rows of that episode)
    ylim = {ep: ylim_for(stats[stats["episode"] == ep], q) for ep in EPISODES}
    for i, (episode, rate) in enumerate(ROW_ORDER):
        for j, area in enumerate(AREAS):
            ax = axes[i, j]
            base = stats[(stats["episode"] == episode) & (stats["area"] == area)]
            ctl = base[base["release_rate"] == 0.0]
            exp = base[base["release_rate"] == rate]
            _draw_series(ax, ctl, q, -DX, CTL_COLOR)
            _draw_series(ax, exp, q, +DX, EPISODE_COLOR.get(episode, INK))
            xlabel = "day" if i == len(ROW_ORDER) - 1 else ""
            _style(ax, f"{episode} · {area} · {rate:g} kt/h",
                   xlabel, "temperature (°C)")
            ax.set_ylim(*ylim[episode])
            ax.set_xlim(0.5, 5.5)
            ax.xaxis.set_major_locator(MultipleLocator(1))

    handles = [
        Line2D([0], [0], color=EPISODE_PALETTE[0], marker="o", linestyle="",
               markersize=6, label="experiment (median)"),
        Line2D([0], [0], color=CTL_COLOR, marker="o", linestyle="",
               markersize=6, label="control (median)"),
        Line2D([0], [0], color=INK, marker="x", linestyle="",
               markersize=7, label="ensemble mean"),
        Line2D([0], [0], color=INK, marker="|", linestyle="",
               markersize=12, label="whisker: ensemble min–max"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"Ensemble {label} vs. day  (experiment colored, control gray)",
        color=INK, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # color denotes episode (line style / gray already distinguishes series)
    for i, ep in enumerate(sorted(load_data()["episode"].unique())):
        EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    stats = build_stats()
    stats.to_csv(OUTPUT_DIR / "daily_ensemble_stats.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "daily_ensemble_stats.xlsx",
                        engine="openpyxl") as xw:
        stats.to_excel(xw, sheet_name="daily_ensemble_stats", index=False)

    outputs = {
        "Tmean": "daily_ensemble_mean_temp.png",
        "Tmax": "daily_ensemble_max_temp.png",
        "Tmin": "daily_ensemble_min_temp.png",
    }
    for q, _, label in QUANTITIES:
        plot_quantity(stats, q, label, OUTPUT_DIR / outputs[q])

    print("Wrote:")
    print("  daily_ensemble_stats.csv")
    print("  daily_ensemble_stats.xlsx")
    for name in outputs.values():
        print(f"  {name}")


if __name__ == "__main__":
    main()
