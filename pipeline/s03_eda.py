#!/usr/bin/env python3
"""
s03_eda.py — premium-ch stage 3: standard premium extraction and EDA figures.

Standard premium definition (proven by count identities in the s02 values
report, consistent with BAG's 'Standardprämie'):

    scope == 'canton'
    is_base_premium  == 1      (base tariff, TAR-BASE)
    is_base_franchise == 1     (CHF 300 adult franchise, FRAST1)
    age_class == 'AKL-ERW'     (adult)
    accident  == 'MIT-UNF'     (with accident cover)

Cantonal aggregate: MEDIAN across insurers and premium regions,
unweighted by enrollment (Versichertenbestand weighting is a declared
extension, current-year insured counts only are published).

Outputs:
    data/processed/standard_premiums.csv   modeling table: year, canton,
                                           region, insurer, premium
    data/processed/canton_median.csv       year, canton, median, n_offers
    figures/f01_canton_trajectories.png    median standard premium per canton
    figures/f02_growth_heatmap.png         YoY growth %, canton x year
    figures/f03_insurer_consolidation.png  insurers offering base tariff
    figures/f04_growth_distribution.png    distribution of cantonal growth per year

Usage (srv-anaconda-01, conda env premium-ch):
    python pipeline/s03_eda.py
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
PARQUET = BASE_DIR / "data" / "processed" / "premiums.parquet"
OUT_STD = BASE_DIR / "data" / "processed" / "standard_premiums.csv"
OUT_MED = BASE_DIR / "data" / "processed" / "canton_median.csv"
FIG_DIR = BASE_DIR / "figures"

DPI = 150
GREY = "#B9BFC7"          # recessive mass
INK = "#1F2430"           # primary text/ink
MUTED = "#6B7280"         # secondary ink
HIGHLIGHT = {             # fixed assignment, entity-stable
    "GE": "#2563EB",
    "ZH": "#059669",
    "AI": "#D97706",
}
NATIONAL = "#DC2626"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GREY, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_standard() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    std = df[
        (df["scope"] == "canton")
        & (df["is_base_premium"].astype(str) == "1")
        & (df["is_base_franchise"].astype(str) == "1")
        & (df["age_class"] == "AKL-ERW")
        & (df["accident"] == "MIT-UNF")
    ].copy()
    std = std[["year", "canton", "region", "insurer", "premium"]]
    std["year"] = std["year"].astype(int)
    log(
        f"standard premium rows: {len(std)} "
        f"({std['year'].min()}-{std['year'].max()}, "
        f"{std['canton'].nunique()} cantons, {std['insurer'].nunique()} insurers)"
    )
    per_year = std.groupby("year").size()
    log("rows per year: " + ", ".join(f"{y}:{n}" for y, n in per_year.items()))
    return std


def cantonal_median(std: pd.DataFrame) -> pd.DataFrame:
    med = (
        std.groupby(["year", "canton"])
        .agg(median_premium=("premium", "median"), n_offers=("premium", "size"))
        .reset_index()
    )
    return med


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_trajectories(med: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    national = med.groupby("year")["median_premium"].median()

    for canton, g in med.groupby("canton"):
        g = g.sort_values("year")
        if canton in HIGHLIGHT:
            continue
        ax.plot(g["year"], g["median_premium"], color=GREY, lw=1.2, zorder=1)

    for canton, color in HIGHLIGHT.items():
        g = med[med["canton"] == canton].sort_values("year")
        ax.plot(g["year"], g["median_premium"], color=color, lw=2.2, zorder=3)
        ax.annotate(
            canton, (g["year"].iloc[-1], g["median_premium"].iloc[-1]),
            xytext=(6, 0), textcoords="offset points",
            color=color, fontsize=10, fontweight="bold", va="center",
        )

    ax.plot(national.index, national.values, color=NATIONAL, lw=2.6,
            ls="--", zorder=4)
    ax.annotate(
        "CH median", (national.index[-1], national.values[-1]),
        xytext=(6, 10), textcoords="offset points",
        color=NATIONAL, fontsize=10, fontweight="bold", va="center",
    )

    style_axes(ax)
    ax.set_title(
        "Standard premium by canton, adult / CHF 300 / base tariff / with accident\n"
        "median across insurers and premium regions, CHF per month",
        fontsize=11, color=INK, loc="left",
    )
    ax.set_xlabel("premium year", color=MUTED, fontsize=9)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f01_canton_trajectories.png", dpi=DPI)
    plt.close(fig)


def fig_growth_heatmap(med: pd.DataFrame) -> None:
    pivot = med.pivot(index="canton", columns="year", values="median_premium")
    growth = (pivot.pct_change(axis=1) * 100).iloc[:, 1:]
    order = pivot.iloc[:, -1].sort_values(ascending=False).index
    growth = growth.loc[order]

    vmax = np.nanmax(np.abs(growth.values))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(growth.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(growth.shape[1]), growth.columns, fontsize=9)
    ax.set_yticks(range(growth.shape[0]), growth.index, fontsize=8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(growth.shape[0]):
        for j in range(growth.shape[1]):
            v = growth.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > 0.6 * vmax else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("YoY growth of median standard premium, %",
                   color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        "Year-over-year growth of the cantonal standard premium\n"
        "cantons ordered by latest premium level",
        fontsize=11, color=INK, loc="left",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f02_growth_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig_consolidation(std: pd.DataFrame) -> None:
    counts = std.groupby("year")["insurer"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color="#2563EB", width=0.7)
    for x, v in counts.items():
        ax.text(x, v + 0.6, str(v), ha="center", fontsize=9, color=INK)
    style_axes(ax)
    ax.set_title(
        "Insurers offering the base tariff (standard premium rows)",
        fontsize=11, color=INK, loc="left",
    )
    ax.set_xlabel("premium year", color=MUTED, fontsize=9)
    ax.set_xticks(list(counts.index))
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f03_insurer_consolidation.png", dpi=DPI)
    plt.close(fig)


def fig_growth_distribution(med: pd.DataFrame) -> None:
    pivot = med.pivot(index="canton", columns="year", values="median_premium")
    growth = (pivot.pct_change(axis=1) * 100).iloc[:, 1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(growth.columns)
    data = [growth[y].dropna().values for y in years]
    bp = ax.boxplot(data, tick_labels=[str(y) for y in years], widths=0.55,
                    patch_artist=True, medianprops=dict(color=NATIONAL, lw=2))
    for box in bp["boxes"]:
        box.set(facecolor="#DBEAFE", edgecolor=MUTED, lw=0.8)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set(color=MUTED, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    style_axes(ax)
    ax.set_title(
        "Distribution of cantonal standard-premium growth (26 cantons per year)\n"
        "box = IQR, red line = median, %",
        fontsize=11, color=INK, loc="left",
    )
    ax.set_xlabel("premium year", color=MUTED, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f04_growth_distribution.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    std = load_standard()
    if std.empty:
        log("standard premium filter returned 0 rows — check s02 output")
        return 1

    std.to_csv(OUT_STD, index=False)
    med = cantonal_median(std)
    med.to_csv(OUT_MED, index=False)
    log(f"written {OUT_STD} and {OUT_MED}")

    # headline numbers for the wiki
    nat = med.groupby("year")["median_premium"].median()
    first, last = nat.index[0], nat.index[-1]
    total = (nat[last] / nat[first] - 1) * 100
    cagr = ((nat[last] / nat[first]) ** (1 / (last - first)) - 1) * 100
    log(f"CH median standard premium: {nat[first]:.2f} CHF ({first}) -> "
        f"{nat[last]:.2f} CHF ({last}), +{total:.1f}% total, {cagr:.2f}%/yr CAGR")
    pivot = med.pivot(index="canton", columns="year", values="median_premium")
    g_last = (pivot[last] / pivot[last - 1] - 1) * 100
    log(f"latest YoY ({last - 1}->{last}): median {g_last.median():.2f}%, "
        f"max {g_last.idxmax()} {g_last.max():.2f}%, "
        f"min {g_last.idxmin()} {g_last.min():.2f}%")

    fig_trajectories(med)
    fig_growth_heatmap(med)
    fig_consolidation(std)
    fig_growth_distribution(med)
    log(f"figures written to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
