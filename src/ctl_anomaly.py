"""Model-free sanity check: control means and release-minus-control anomalies.

Works directly from the ensemble members (no regression):

1. For each (episode, area, hour): mean and standard error of the 3 ctl members.
2. For each (episode, area, release_case, hour): the anomaly relative to the
   matching ctl, mean(case) - mean(ctl), with uncertainty propagated two ways:
     - paired   SE = std(case_i - ctl_i, ddof=1) / sqrt(n)   [members matched]
     - unpaired SE = sqrt(SE_case^2 + SE_ctl^2)
   The ensemble members e1/e2/e3 are matched across scenarios, so the paired SE
   is the proper one; the unpaired SE is reported for comparison.

Outputs (data/output/):
    ctl_mean_se.csv                  control mean/SE per (episode, area, hour)
    t2_anomalies.csv / .xlsx         anomalies (long) + a ctl sheet
    ctl_mean_se.png                  control T2 vs hour, 95% band
    anomaly_r1.png / _r10 / _r100    anomaly vs hour per release rate, 95% band
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fit_t2  # sibling module
from fit_t2 import load_data, plot_coefficient, OUTPUT_DIR, EPISODE_PALETTE

N_KEY = ["episode", "area", "hour"]


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def compute(df: pd.DataFrame):
    # --- control mean / SE per (episode, area, hour) ------------------------
    ctl = df[df["scenario"] == "ctl"]
    ctl_summary = (
        ctl.groupby(N_KEY)["T2"]
        .agg(ctl_mean="mean", ctl_se=sem, n_ctl="count")
        .reset_index()
    )

    # --- paired anomalies: attach the matching ctl member to each case row ---
    ctl_members = ctl[N_KEY + ["ens", "T2"]].rename(columns={"T2": "T2_ctl"})
    cases = df[df["scenario"] != "ctl"].merge(ctl_members, on=N_KEY + ["ens"], how="left")
    cases["d"] = cases["T2"] - cases["T2_ctl"]

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        se_case = sem(g["T2"])
        se_ctl = sem(g["T2_ctl"])
        return pd.Series({
            "release_rate": g["release_rate"].iloc[0],
            "n": n,
            "dT2": float(g["d"].mean()),          # = mean(case) - mean(ctl)
            "se_paired": sem(g["d"]),
            "se_unpaired": float(np.sqrt(se_case ** 2 + se_ctl ** 2)),
            "mean_case": float(g["T2"].mean()), "se_case": se_case,
            "mean_ctl": float(g["T2_ctl"].mean()), "se_ctl": se_ctl,
        })

    anomalies = (
        cases.groupby(["episode", "area", "scenario", "hour"], group_keys=True)
        .apply(agg, include_groups=False)
        .reset_index()
        .sort_values(["episode", "area", "release_rate", "hour"])
        .reset_index(drop=True)
    )
    return ctl_summary, anomalies


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # color per episode (line style encodes area) for the shared plot helper
    for i, ep in enumerate(sorted(df["episode"].unique())):
        fit_t2.EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    ctl_summary, anomalies = compute(df)

    # --- tables -------------------------------------------------------------
    ctl_csv = OUTPUT_DIR / "ctl_mean_se.csv"
    anom_csv = OUTPUT_DIR / "t2_anomalies.csv"
    xlsx = OUTPUT_DIR / "t2_anomalies.xlsx"
    ctl_summary.to_csv(ctl_csv, index=False)
    anomalies.to_csv(anom_csv, index=False)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        anomalies.to_excel(xw, sheet_name="anomalies", index=False)
        ctl_summary.to_excel(xw, sheet_name="ctl", index=False)

    # --- plots --------------------------------------------------------------
    plot_coefficient(ctl_summary, "ctl_mean", "ctl T2 (K)",
                     "Control T2: mean +/- 95% CI vs. hour",
                     OUTPUT_DIR / "ctl_mean_se.png", se_column="ctl_se")

    for rate in sorted(anomalies["release_rate"].unique()):
        sub = anomalies[anomalies["release_rate"] == rate]
        plot_coefficient(
            sub, "dT2", "T2 anomaly (K)",
            f"T2 anomaly (release - ctl), {rate:g} kt/h  (band: 95% CI, paired)",
            OUTPUT_DIR / f"anomaly_r{rate:g}.png", se_column="se_paired",
        )

    # --- cross-check vs. the shared-beta deltaT2_10 -------------------------
    sb_path = OUTPUT_DIR / "t2_sharedbeta_fits.csv"
    print("Control mean range: "
          f"{ctl_summary['ctl_mean'].min():.2f}-{ctl_summary['ctl_mean'].max():.2f} K")
    print("Median paired SE by release rate:")
    print(anomalies.groupby("release_rate")["se_paired"].median().round(3).to_string())
    print("Paired vs unpaired SE (median ratio): "
          f"{(anomalies['se_paired'] / anomalies['se_unpaired']).median():.3f}")
    if sb_path.exists():
        sb = pd.read_csv(sb_path)
        sb["episode"] = sb["episode"].astype(str)
        r10 = anomalies[anomalies["release_rate"] == 10][
            ["episode", "area", "hour", "dT2"]]
        chk = r10.merge(sb[["episode", "area", "hour", "deltaT2_10"]],
                        on=["episode", "area", "hour"])
        rmsd = float(np.sqrt(((chk["dT2"] - chk["deltaT2_10"]) ** 2).mean()))
        print(f"Cross-check r10 anomaly vs shared-beta deltaT2_10: RMSD = {rmsd:.4f} K")

    print(f"\nWrote:\n  {ctl_csv}\n  {anom_csv}\n  {xlsx}")
    print("  ctl_mean_se.png, anomaly_r1.png, anomaly_r10.png, anomaly_r100.png")


if __name__ == "__main__":
    main()
