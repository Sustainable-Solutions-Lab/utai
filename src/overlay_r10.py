"""Overlay the raw 10 kt/h anomaly on the shared-beta deltaT2_10, per combo.

Visual cross-check of the shared-beta fit against the model-free anomaly. For
each (episode, area) it draws, vs. hour:
  - fitted deltaT2_10 (shared beta) with its 95% band (deltaT2_10_se_total)
  - raw r10 - ctl anomaly with its 95% paired band (se_paired)

Reads the outputs of fit_t2_shared_beta.py and ctl_anomaly.py.

Outputs (data/output/):
    deltaT2_10_vs_anomaly_r10.png
    r10_vs_deltaT2_10.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit_t2 import OUTPUT_DIR, SURFACE, INK, AXIS, style_axes, day_ticks

FIT_COLOR = "#2a78d6"   # shared-beta deltaT2_10
RAW_COLOR = "#eb6834"   # raw r10 anomaly
Z = 1.96


def main():
    sb_path = OUTPUT_DIR / "t2_sharedbeta_fits.csv"
    an_path = OUTPUT_DIR / "t2_anomalies.csv"
    for p, script in [(sb_path, "fit_t2_shared_beta.py"), (an_path, "ctl_anomaly.py")]:
        if not p.exists():
            sys.exit(f"missing {p} - run src/{script} first")

    sb = pd.read_csv(sb_path)
    sb["episode"] = sb["episode"].astype(str)
    an = pd.read_csv(an_path)
    an["episode"] = an["episode"].astype(str)
    r10 = an[an["release_rate"] == 10]

    merged = r10.merge(
        sb[["episode", "area", "hour", "deltaT2_10", "deltaT2_10_se_total"]],
        on=["episode", "area", "hour"],
    ).sort_values(["episode", "area", "hour"])
    merged["diff"] = merged["dT2"] - merged["deltaT2_10"]
    out = merged[[
        "episode", "area", "hour",
        "dT2", "se_paired", "deltaT2_10", "deltaT2_10_se_total", "diff",
    ]].rename(columns={"dT2": "anomaly_r10", "se_paired": "anomaly_r10_se"})
    out.to_csv(OUTPUT_DIR / "r10_vs_deltaT2_10.csv", index=False)

    episodes = sorted(merged["episode"].unique())
    areas = sorted(merged["area"].unique())
    fig, axes = plt.subplots(len(episodes), len(areas), figsize=(12, 8),
                             sharex=True, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    for i, ep in enumerate(episodes):
        for j, ar in enumerate(areas):
            ax = axes[i, j]
            g = merged[(merged["episode"] == ep) & (merged["area"] == ar)]
            ax.axhline(0, color=AXIS, linewidth=1, zorder=0)

            se_f = g["deltaT2_10_se_total"].to_numpy(float)
            ax.fill_between(g["hour"], g["deltaT2_10"] - Z * se_f,
                            g["deltaT2_10"] + Z * se_f,
                            color=FIT_COLOR, alpha=0.15, linewidth=0)
            ax.plot(g["hour"], g["deltaT2_10"], color=FIT_COLOR, linewidth=2,
                    label="shared-beta deltaT2_10")

            se_r = g["se_paired"].to_numpy(float)
            ax.fill_between(g["hour"], g["dT2"] - Z * se_r, g["dT2"] + Z * se_r,
                            color=RAW_COLOR, alpha=0.15, linewidth=0)
            ax.plot(g["hour"], g["dT2"], color=RAW_COLOR, linewidth=1.5,
                    linestyle="--", label="raw r10 - ctl anomaly")

            rmsd = float(np.sqrt((g["diff"] ** 2).mean()))
            style_axes(ax, f"{ar} · {ep}   (RMSD {rmsd:.2f} K)", "deltaT2_10 (K)")
            day_ticks(ax)
            if i < len(episodes) - 1:
                ax.set_xlabel("")

    handles = [
        plt.Line2D([0], [0], color=FIT_COLOR, linewidth=2,
                   label="shared-beta deltaT2_10"),
        plt.Line2D([0], [0], color=RAW_COLOR, linewidth=1.5, linestyle="--",
                   label="raw r10 - ctl anomaly"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10, labelcolor=INK,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("10 kt/h: shared-beta fit vs. model-free anomaly  (bands: 95% CI)",
                 color=INK, fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deltaT2_10_vs_anomaly_r10.png",
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    print("Per-combo RMSD (raw r10 anomaly vs shared-beta deltaT2_10):")
    print(merged.groupby(["episode", "area"])
          .apply(lambda g: np.sqrt((g["diff"] ** 2).mean()), include_groups=False)
          .round(3).to_string())
    print(f"\nWrote:\n  {OUTPUT_DIR / 'deltaT2_10_vs_anomaly_r10.png'}"
          f"\n  {OUTPUT_DIR / 'r10_vs_deltaT2_10.csv'}")


if __name__ == "__main__":
    main()
