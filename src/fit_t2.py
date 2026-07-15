"""Hourly dose-response regression of T2 vs. release rate.

For each (episode, evaluation area) and each hour h, fit across the 12 samples
(4 release cases x 3 ensemble members):

    T2 = T2_0 + deltaT2_10 * (release_rate / 10) ** beta

where release_rate is in kt/h: ctl -> 0, 1000_5x5 -> 1, 10000_5x5 -> 10,
100000_5x5 -> 100.  At release_rate = 0 the power term is 0, so T2_0 is the
control temperature; deltaT2_10 is the perturbation at 10 kt/h; beta sets the
dose-response curvature.

Outputs (to data/output/):
    t2_hourly_fits.csv   - best-fit coefficients + ancillary stats per fit
    t2_hourly_fits.xlsx  - same table
    T2_0.png, deltaT2_10.png, beta.png - coefficient vs. hour, 4 combos each
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "input" / "T2_summary.csv"
OUTPUT_DIR = ROOT / "data" / "output"

# Release cases we expect and their numeric release rate (kt/h).
RELEASE_RATES = [0.0, 1.0, 10.0, 100.0]

# --- chart chrome (from the dataviz reference palette, light mode) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
AREA_STYLE = {"city": "--", "region": "-"}  # dashed city / solid region
EPISODE_PALETTE = ["#2a78d6", "#eb6834", "#008300", "#4a3aa7"]  # CVD-safe hues
EPISODE_COLOR = {}  # filled in at runtime from the data (color denotes episode)


def release_rate(scenario: str) -> float:
    """Map a scenario label to release rate in kt/h.

    'ctl' -> 0; otherwise the number before the first '_' divided by 1000
    (1000_5x5 -> 1, 10000_5x5 -> 10, 100000_5x5 -> 100).
    """
    if scenario == "ctl":
        return 0.0
    return float(scenario.split("_")[0]) / 1000.0


def model(rr, t2_0, dt10, beta):
    """T2 = t2_0 + dt10 * (rr/10)**beta, with the rr=0 term pinned to 0."""
    rr = np.asarray(rr, dtype=float)
    ratio = np.where(rr > 0, rr / 10.0, 1.0)  # avoid 0**beta warnings
    term = np.where(rr > 0, np.power(ratio, beta), 0.0)
    return t2_0 + dt10 * term


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    # Source column is misspelled 'evaluatino_area'; normalize to 'area'.
    df = df.rename(columns={"evaluatino_area": "area", "value": "T2"})
    df["episode"] = df["episode"].astype(str)
    df["hour"] = df["hour_index"].astype(int)
    df["release_rate"] = df["scenario"].map(release_rate)
    return df


def fit_one(sub: pd.DataFrame) -> dict:
    """Fit the model to one (episode, area, hour) group of 12 samples."""
    x = sub["release_rate"].to_numpy(dtype=float)
    y = sub["T2"].to_numpy(dtype=float)

    # per-release-rate means (ancillary + used for initial guess)
    means = {r: y[x == r].mean() if np.any(x == r) else np.nan for r in RELEASE_RATES}

    t0_0 = means[0.0] if np.isfinite(means[0.0]) else y.mean()
    d0 = (means[10.0] - t0_0) if np.isfinite(means[10.0]) else -0.1
    p0 = [t0_0, d0 if d0 != 0 else -0.1, 1.0]
    bounds = ([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 10.0])

    rec = {
        "n_points": len(y),
        "mean_r0": means[0.0],
        "mean_r1": means[1.0],
        "mean_r10": means[10.0],
        "mean_r100": means[100.0],
    }
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=40000)
        perr = np.sqrt(np.diag(pcov))
        yhat = model(x, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rec.update(
            T2_0=popt[0],
            deltaT2_10=popt[1],
            beta=popt[2],
            T2_0_se=perr[0],
            deltaT2_10_se=perr[1],
            beta_se=perr[2],
            rmse=float(np.sqrt(ss_res / len(y))),
            r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            converged=True,
        )
    except Exception as exc:  # noqa: BLE001 - record failure, keep going
        rec.update(
            T2_0=np.nan, deltaT2_10=np.nan, beta=np.nan,
            T2_0_se=np.nan, deltaT2_10_se=np.nan, beta_se=np.nan,
            rmse=np.nan, r2=np.nan, converged=False, error=str(exc),
        )
    return rec


def run_fits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (episode, area, hour), sub in df.groupby(["episode", "area", "hour"]):
        rec = {"episode": episode, "area": area, "hour": hour}
        rec.update(fit_one(sub))
        rows.append(rec)
    cols = [
        "episode", "area", "hour", "n_points",
        "T2_0", "deltaT2_10", "beta",
        "T2_0_se", "deltaT2_10_se", "beta_se",
        "r2", "rmse",
        "mean_r0", "mean_r1", "mean_r10", "mean_r100",
        "converged",
    ]
    out = pd.DataFrame(rows)
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].sort_values(["episode", "area", "hour"]).reset_index(drop=True)


def day_ticks(ax):
    """Hour axis with day-aligned ticks: major every 24 h, minor every 12 h."""
    ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.xaxis.set_minor_locator(MultipleLocator(12))
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)


def style_axes(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=10)
    ax.set_xlabel("hour", color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_coefficient(fits: pd.DataFrame, column: str, ylabel: str, title: str,
                     path: Path, se_column: str | None = None, z: float = 1.96):
    """Coefficient vs. hour, one line per (episode, area).

    If se_column is given (and present), a shaded +/- z*se band is drawn around
    each line (z=1.96 -> ~95% interval).
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for area in sorted(fits["area"].unique()):
        for episode in sorted(fits["episode"].unique()):
            sub = fits[(fits["area"] == area) & (fits["episode"] == episode)].sort_values("hour")
            color = EPISODE_COLOR.get(episode, INK)
            if se_column and se_column in sub.columns:
                se = sub[se_column].to_numpy(float)
                ax.fill_between(
                    sub["hour"], sub[column] - z * se, sub[column] + z * se,
                    color=color, alpha=0.13, linewidth=0,
                )
            ax.plot(
                sub["hour"], sub[column],
                color=color, linestyle=AREA_STYLE.get(area, "-"),
                linewidth=2, label=f"{area} · {episode}",
            )
    style_axes(ax, title, ylabel)
    day_ticks(ax)

    handles = [
        Line2D([0], [0], color=EPISODE_COLOR.get(episode, INK),
               linestyle=AREA_STYLE.get(area, "-"), linewidth=2,
               label=f"{area} · {episode}")
        for area in sorted(fits["area"].unique())
        for episode in sorted(fits["episode"].unique())
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # assign a color per episode (line style already encodes area)
    for i, ep in enumerate(sorted(df["episode"].unique())):
        EPISODE_COLOR[ep] = EPISODE_PALETTE[i % len(EPISODE_PALETTE)]

    fits = run_fits(df)

    csv_path = OUTPUT_DIR / "t2_hourly_fits.csv"
    xlsx_path = OUTPUT_DIR / "t2_hourly_fits.xlsx"
    fits.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        fits.to_excel(xw, sheet_name="fits", index=False)

    plot_coefficient(fits, "T2_0", "T2_0 (K)",
                     "Control temperature T2_0 vs. hour  (band: 95% CI)",
                     OUTPUT_DIR / "T2_0.png", se_column="T2_0_se")
    plot_coefficient(fits, "deltaT2_10", "deltaT2_10 (K)",
                     "Perturbation at 10 kt/h (deltaT2_10) vs. hour  (band: 95% CI)",
                     OUTPUT_DIR / "deltaT2_10.png", se_column="deltaT2_10_se")
    plot_coefficient(fits, "beta", "beta",
                     "Dose-response exponent beta vs. hour  (band: 95% CI)",
                     OUTPUT_DIR / "beta.png", se_column="beta_se")

    n_bad = int((~fits["converged"]).sum())
    print(f"Fits: {len(fits)} groups  |  non-converged: {n_bad}")
    print(f"Median R^2: {fits['r2'].median():.4f}")
    print(f"Wrote:\n  {csv_path}\n  {xlsx_path}")
    print("  T2_0.png, deltaT2_10.png, beta.png")


if __name__ == "__main__":
    main()
