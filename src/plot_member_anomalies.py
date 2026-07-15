"""Plot the per-member paired anomaly T2_exp - T2_ctl vs hour for one case.

One line per ensemble member (e1/e2/e3), plus their mean, for a single
(episode, area, release_rate). Useful for inspecting member spread at a given
dose -- e.g. the region 240527 @ 10 kt/h case that fits worst.

Usage:  python src/plot_member_anomalies.py [episode] [area] [release_rate]
        defaults: 240527 region 10

Output: data/output/member_anomalies_<episode>_<area>_r<rate>.png
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from fit_t2 import load_data, OUTPUT_DIR, SURFACE, INK, AXIS, style_axes, day_ticks
from fit_t2_shared_beta_anomaly import build_anomalies

MEMBER_COLOR = {"e1": "#2a78d6", "e2": "#008300", "e3": "#eb6834"}


def main():
    episode = sys.argv[1] if len(sys.argv) > 1 else "240527"
    area = sys.argv[2] if len(sys.argv) > 2 else "region"
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    df = load_data()
    an = build_anomalies(df)
    an["episode"] = an["episode"].astype(str)
    g = an[(an["episode"] == episode) & (an["area"] == area)
           & (an["release_rate"] == rate)]
    if g.empty:
        sys.exit(f"no data for episode={episode} area={area} rate={rate:g}")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=0)

    for ens in sorted(g["ens"].unique()):
        sub = g[g["ens"] == ens].sort_values("hour")
        ax.plot(sub["hour"], sub["d"], color=MEMBER_COLOR.get(ens, INK),
                linewidth=1.5, label=ens)

    mean = g.groupby("hour")["d"].mean().reset_index()
    ax.plot(mean["hour"], mean["d"], color=INK, linewidth=2, linestyle="--",
            label="member mean")

    style_axes(ax, f"T2_exp - T2_ctl per member — {area} · {episode} @ {rate:g} kt/h",
               "T2_exp - T2_ctl (K)")
    day_ticks(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=4, loc="best")
    fig.tight_layout()

    out = OUTPUT_DIR / f"member_anomalies_{episode}_{area}_r{rate:g}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote:\n  {out}")


if __name__ == "__main__":
    main()
