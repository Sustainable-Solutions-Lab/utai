"""Predicted-vs-observed paired anomaly, parametric in hour. 12 panels, one page.

For each (episode, area, release_rate) the curve is parameterized by hour h:
  x(h) = observed paired anomaly, mean over members of (T2_exp - T2_ctl) at hour h
  y(h) = predicted by the fit, deltaT2_10(h) * dose_factor(release_rate)
Points are colored by hour and joined in hour order, so the 5-day trajectory is
visible; the 1:1 line is the perfect-prediction reference. Because the fit shares
one shape parameter across the three release rates, a single rate's curve need not
lie exactly on 1:1 -- departures show where one dose pulls against the others.

Usage:  python src/scatter_pred_obs.py [power|saturation]   (default: power)
  power       -> reads t2_anomaly_fit.csv     (fit_t2_shared_beta_anomaly.py)
  saturation  -> reads t2_saturation_fit.csv  (fit_t2_saturation_anomaly.py)

Output: data/output/scatter_pred_vs_obs_{power,saturation}.png
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from fit_t2 import load_data, OUTPUT_DIR, SURFACE, INK, INK_MUTED, AXIS, GRID
from fit_t2_shared_beta_anomaly import build_anomalies
from fit_t2_saturation_anomaly import dose_term as sat_dose_term

CMAP = "viridis"

MODELS = {
    "power": {
        "csv": "t2_anomaly_fit.csv", "param": "beta",
        "factor": lambda rate, p: (rate / 10.0) ** p,
        "out": "scatter_pred_vs_obs_power.png", "label": "power law",
    },
    "saturation": {
        "csv": "t2_saturation_fit.csv", "param": "release_scale",
        "factor": lambda rate, p: sat_dose_term(np.asarray(rate, float), p),
        "out": "scatter_pred_vs_obs_saturation.png", "label": "exp saturation",
    },
}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "power"
    if model not in MODELS:
        sys.exit(f"unknown model '{model}'; choose from {list(MODELS)}")
    cfg = MODELS[model]

    fit_path = OUTPUT_DIR / cfg["csv"]
    if not fit_path.exists():
        sys.exit(f"missing {fit_path} - run the corresponding fit script first")

    df = load_data()
    fit = pd.read_csv(fit_path)
    fit["episode"] = fit["episode"].astype(str)
    an = build_anomalies(df)
    an["episode"] = an["episode"].astype(str)

    # member-mean observed anomaly per (episode, area, rate, hour)
    obs = (an.groupby(["episode", "area", "release_rate", "hour"])["d"]
           .mean().reset_index().rename(columns={"d": "obs"}))
    m = obs.merge(fit[["episode", "area", "hour", "deltaT2_10", cfg["param"]]],
                  on=["episode", "area", "hour"])
    # dose factor is scalar per (combo, rate) via the shared shape parameter
    m["pred"] = m["deltaT2_10"] * m.apply(
        lambda r: cfg["factor"](r["release_rate"], r[cfg["param"]]), axis=1)

    episodes = sorted(m["episode"].unique())
    areas = sorted(m["area"].unique())
    combos = [(ep, ar) for ep in episodes for ar in areas]   # 4 rows
    rates = sorted(m["release_rate"].unique())               # 3 cols: 1, 10, 100
    hmin, hmax = int(m["hour"].min()), int(m["hour"].max())
    norm = Normalize(hmin, hmax)

    fig, axes = plt.subplots(len(combos), len(rates), figsize=(11, 14),
                             dpi=150, layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for r, (ep, ar) in enumerate(combos):
        for c, rate in enumerate(rates):
            ax = axes[r, c]
            g = m[(m["episode"] == ep) & (m["area"] == ar)
                  & (m["release_rate"] == rate)].sort_values("hour")
            x = g["obs"].to_numpy(float)
            y = g["pred"].to_numpy(float)

            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            lo, hi = lo - pad, hi + pad

            ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=0)
            ax.plot(x, y, color=INK_MUTED, linewidth=0.6, alpha=0.35, zorder=1)
            ax.scatter(x, y, c=g["hour"], cmap=CMAP, norm=norm, s=16,
                       edgecolor="none", zorder=2)

            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            rr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else np.nan
            ax.text(0.04, 0.96, f"RMSE {rmse:.2f} K\nr {rr:.2f}",
                    transform=ax.transAxes, ha="left", va="top",
                    color=INK_MUTED, fontsize=8)

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

            if r == 0:
                ax.set_title(f"{rate:g} kt/h", color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{ar} · {ep}\n\npredicted (K)",
                              color=INK_MUTED, fontsize=9)
            if r == len(combos) - 1:
                ax.set_xlabel("observed  T_exp - T_ctl (K)",
                              color=INK_MUTED, fontsize=9)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02, aspect=40)
    cbar.set_label("hour", color=INK_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.suptitle(f"Predicted vs. observed paired anomaly, parametric in hour  "
                 f"({cfg['label']}; line = 1:1)", color=INK, fontsize=14)

    out = OUTPUT_DIR / cfg["out"]
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote:\n  {out}")


if __name__ == "__main__":
    main()
