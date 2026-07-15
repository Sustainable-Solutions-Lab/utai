"""Plot the per-member paired anomaly T2_exp - T2_ctl vs hour for one case.

One line per ensemble member (e1/e2/e3) and their mean, for a single
(episode, area, release_rate), with the fitted T2_scale(h) overlaid. Useful for
inspecting member spread at a given dose -- e.g. the region 240527 @ 10 kt/h case
that fits worst. Because T2_scale is the perturbation at 10 kt/h, at rate = 10 the
overlay is exactly the fit's prediction; at other rates the prediction would be
T2_scale * dose_factor(rate), but T2_scale itself is still shown for reference.

Usage:  python src/plot_member_anomalies.py [episode] [area] [release_rate] [model]
        defaults: 240527 region 10 power        (model: power | saturation)

Output: data/output/member_anomalies_<episode>_<area>_r<rate>_<model>.png
"""

from __future__ import annotations

import sys

import pandas as pd
import matplotlib.pyplot as plt

from fit_t2 import load_data, OUTPUT_DIR, SURFACE, INK, AXIS, style_axes, day_ticks
from fit_t2_shared_beta_anomaly import build_anomalies

MEMBER_COLOR = {"e1": "#2a78d6", "e2": "#008300", "e3": "#eb6834"}
T2SCALE_COLOR = "#4a3aa7"  # violet, distinct from members and the mean
FIT_CSV = {"power": "t2_anomaly_fit.csv", "saturation": "t2_saturation_fit.csv"}


def main():
    episode = sys.argv[1] if len(sys.argv) > 1 else "240527"
    area = sys.argv[2] if len(sys.argv) > 2 else "region"
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    model = sys.argv[4] if len(sys.argv) > 4 else "power"
    if model not in FIT_CSV:
        sys.exit(f"unknown model '{model}'; choose from {list(FIT_CSV)}")

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

    # overlay fitted T2_scale(h) (= deltaT2_10 column) from the chosen model
    fit_path = OUTPUT_DIR / FIT_CSV[model]
    if not fit_path.exists():
        sys.exit(f"missing {fit_path} - run the {model} fit script first")
    fit = pd.read_csv(fit_path)
    fit["episode"] = fit["episode"].astype(str)
    fg = fit[(fit["episode"] == episode) & (fit["area"] == area)].sort_values("hour")
    ax.plot(fg["hour"], fg["deltaT2_10"], color=T2SCALE_COLOR, linewidth=2.5,
            label=f"T2_scale ({model} fit)")

    style_axes(ax, f"T2_exp - T2_ctl per member — {area} · {episode} @ {rate:g} kt/h",
               "T2_exp - T2_ctl (K)")
    day_ticks(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=5, loc="best")
    fig.tight_layout()

    out = OUTPUT_DIR / f"member_anomalies_{episode}_{area}_r{rate:g}_{model}.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote:\n  {out}")


if __name__ == "__main__":
    main()
