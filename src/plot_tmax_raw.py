"""Plot the raw daily-max reductions from tmax_reductions.csv.

Two 4-panel figures (one panel per episode x area), showing every
(member, day, dose) reduction point:

  tmax_raw_vs_rate.png  Tmax_red vs release_rate (log x), colored by day,
                        with the per-rate mean overlaid.
  tmax_raw_vs_day.png   Tmax_red vs day, colored by release rate, with the
                        per-(rate, day) mean line.

Reads data/output/tmax_reductions.csv (run fit_tmax_saturation.py first).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator

from fit_t2 import OUTPUT_DIR, SURFACE, INK, INK_MUTED, GRID, AXIS

CMAP_DAY = "viridis"
RATE_COLOR = {1.0: "#2a78d6", 10.0: "#eb6834", 100.0: "#008300"}


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


def main():
    path = OUTPUT_DIR / "tmax_reductions.csv"
    if not path.exists():
        sys.exit(f"missing {path} - run src/fit_tmax_saturation.py first")
    red = pd.read_csv(path)
    red["episode"] = red["episode"].astype(str)

    episodes = sorted(red["episode"].unique())
    areas = sorted(red["area"].unique())
    combos = [(ep, ar) for ep in episodes for ar in areas]
    rates = sorted(red["release_rate"].unique())
    days = sorted(red["day"].unique())
    norm = Normalize(min(days), max(days))

    # ---- Figure 1: reduction vs release rate, colored by day -------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ep, ar) in zip(axes.ravel(), combos):
        g = red[(red["episode"] == ep) & (red["area"] == ar)]
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        ax.scatter(g["release_rate"], g["Tmax_red"], c=g["day"], cmap=CMAP_DAY,
                   norm=norm, s=30, alpha=0.75, edgecolor="none", zorder=2)
        m = g.groupby("release_rate")["Tmax_red"].mean()
        ax.plot(m.index, m.values, color=INK, marker="o", markersize=6,
                linewidth=1.5, zorder=3, label="mean")
        ax.set_xscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:g}" for r in rates])
        _style(ax, f"{ar} · {ep}", "release rate (kt/h)",
               "Tmax_red = ctl - exp (K)")
    sm = ScalarMappable(norm=norm, cmap=CMAP_DAY)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=30,
                        ticks=days)
    cbar.set_label("day", color=INK_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle("Daily-max reduction vs. release rate  (points: member x day; "
                 "black: mean)", color=INK, fontsize=13)
    fig.savefig(OUTPUT_DIR / "tmax_raw_vs_rate.png", facecolor=SURFACE)
    plt.close(fig)

    # ---- Figure 2: reduction vs day, colored by release rate -------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (ep, ar) in zip(axes.ravel(), combos):
        g = red[(red["episode"] == ep) & (red["area"] == ar)]
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        for r in rates:
            gr = g[g["release_rate"] == r]
            color = RATE_COLOR.get(r, INK)
            ax.scatter(gr["day"], gr["Tmax_red"], color=color, s=26, alpha=0.55,
                       edgecolor="none", zorder=2)
            mm = gr.groupby("day")["Tmax_red"].mean()
            ax.plot(mm.index, mm.values, color=color, linewidth=2, marker="o",
                    markersize=4, zorder=3, label=f"{r:g} kt/h")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        _style(ax, f"{ar} · {ep}", "day", "Tmax_red = ctl - exp (K)")
    axes.ravel()[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")
    fig.suptitle("Daily-max reduction vs. day  (points: members; lines: per-rate mean)",
                 color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tmax_raw_vs_day.png", facecolor=SURFACE)
    plt.close(fig)

    print("Wrote:\n  tmax_raw_vs_rate.png\n  tmax_raw_vs_day.png")


if __name__ == "__main__":
    main()
