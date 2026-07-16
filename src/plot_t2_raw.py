"""Raw T2 traces: experiment vs. its paired control, per ensemble member.

Four panels, one per (episode, area).  Each panel has 6 lines: the three
ensemble members' experimental T2 at a single release rate (dashed, one color
per member) and the same members' control T2 (solid, same colors).

Usage:
    python src/plot_t2_raw.py [release_rate]   # default 10; 0 is the control

Outputs (to data/output/):
    t2_raw_members_r<rate>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fit_t2 import (
    INK,
    OUTPUT_DIR,
    SURFACE,
    day_ticks,
    load_data,
    style_axes,
)

# color = ensemble member (line style encodes exp vs. ctl)
MEMBER_PALETTE = ["#2a78d6", "#eb6834", "#008300", "#4a3aa7"]
MEMBER_COLOR: dict[str, str] = {}


def plot_raw(df, rate: float, path: Path):
    episodes = sorted(df["episode"].unique())
    areas = sorted(df["area"].unique())
    members = sorted(df["ens"].unique())

    fig, axes = plt.subplots(
        len(episodes), len(areas), figsize=(13, 8), dpi=150, sharex=True
    )
    fig.patch.set_facecolor(SURFACE)

    for i, episode in enumerate(episodes):
        for j, area in enumerate(areas):
            ax = axes[i, j]
            panel = df[(df["episode"] == episode) & (df["area"] == area)]
            for member in members:
                color = MEMBER_COLOR[member]
                for rr, style, width in ((rate, "--", 1.0), (0.0, "-", 0.8)):
                    sub = panel[
                        (panel["ens"] == member) & (panel["release_rate"] == rr)
                    ].sort_values("hour")
                    ax.plot(
                        sub["hour"],
                        sub["T2"],
                        color=color,
                        linestyle=style,
                        linewidth=width,
                    )
            style_axes(ax, f"{episode} · {area}", "T2 (K)")
            day_ticks(ax)

    handles = [
        Line2D([0], [0], color=MEMBER_COLOR[m], linestyle="--", linewidth=1.4,
               label=f"{m} · {rate:g} kt/h")
        for m in members
    ] + [
        Line2D([0], [0], color=MEMBER_COLOR[m], linestyle="-", linewidth=1.2,
               label=f"{m} · control")
        for m in members
    ]
    fig.legend(
        handles=handles, frameon=False, fontsize=9, labelcolor=INK,
        loc="lower center", ncol=len(members) * 2,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Raw T2 vs. hour — experiment at {rate:g} kt/h (dashed) vs. paired control (solid)",
        color=INK, fontsize=14, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    available = sorted(df["release_rate"].unique())
    if rate not in available:
        raise SystemExit(f"release_rate {rate:g} not in data; have {available}")

    for i, member in enumerate(sorted(df["ens"].unique())):
        MEMBER_COLOR[member] = MEMBER_PALETTE[i % len(MEMBER_PALETTE)]

    path = OUTPUT_DIR / f"t2_raw_members_r{rate:g}.png"
    plot_raw(df, rate, path)
    print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
