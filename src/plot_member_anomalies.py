"""Plot the per-member paired anomaly T2_exp - T2_ctl vs hour for one case.

One line per ensemble member (e1/e2/e3) and their mean, for a single
(episode, area, release_rate), with the model PREDICTION at that release rate
overlaid. The prediction is T2_scale(h) * dose_factor(release_rate): at 10 kt/h
the factor is 1 (so the overlay is T2_scale itself); at 100 kt/h it is scaled up
by the shared dose-response so predicted and observed are on the same footing.

Usage:  python src/plot_member_anomalies.py [episode] [area] [release_rate] [model]
        defaults: 240527 region 10 power        (model: power | saturation)

Output: data/output/member_anomalies_<episode>_<area>_r<rate>_<model>.png
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit_t2 import load_data, OUTPUT_DIR, SURFACE, INK, AXIS, style_axes, day_ticks
from fit_t2_shared_beta_anomaly import build_anomalies
from fit_t2_saturation_anomaly import dose_term as sat_dose_term

MEMBER_COLOR = {"e1": "#2a78d6", "e2": "#008300", "e3": "#eb6834"}
PRED_COLOR = "#4a3aa7"  # violet, distinct from members and the mean
MODELS = {
    "power": {"csv": "t2_anomaly_fit.csv", "param": "beta",
              "factor": lambda rate, p: (rate / 10.0) ** p},
    "saturation": {"csv": "t2_saturation_fit.csv", "param": "release_scale",
                   "factor": lambda rate, p: float(sat_dose_term(np.array([rate]), p)[0])},
}


def main():
    episode = sys.argv[1] if len(sys.argv) > 1 else "240527"
    area = sys.argv[2] if len(sys.argv) > 2 else "region"
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    model = sys.argv[4] if len(sys.argv) > 4 else "power"
    if model not in MODELS:
        sys.exit(f"unknown model '{model}'; choose from {list(MODELS)}")
    cfg = MODELS[model]

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

    # overlay the model prediction at this release rate:
    # T2_scale(h) * dose_factor(rate).  factor = 1 at 10 kt/h.
    fit_path = OUTPUT_DIR / cfg["csv"]
    if not fit_path.exists():
        sys.exit(f"missing {fit_path} - run the {model} fit script first")
    fit = pd.read_csv(fit_path)
    fit["episode"] = fit["episode"].astype(str)
    fg = fit[(fit["episode"] == episode) & (fit["area"] == area)].sort_values("hour")
    factor = cfg["factor"](rate, float(fg[cfg["param"]].iloc[0]))
    ax.plot(fg["hour"], fg["deltaT2_10"] * factor, color=PRED_COLOR, linewidth=2.5,
            label=f"predicted @ {rate:g} kt/h ({model})")

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
