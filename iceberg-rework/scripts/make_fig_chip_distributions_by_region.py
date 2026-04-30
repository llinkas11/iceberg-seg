"""
make_fig_chip_distributions_by_region.py: Chip count by month and ERA5
wind/temperature histograms colored by region (KQ vs SK). Mirrors Fisser 2025
Fig. 8 + Fig. 9 layout, scoped to our two southeast Greenland fjords.

Reads:
  reference/met_data.csv (chip_stem, region, sza_bin, date, wind_speed_10m, temp_2m, source)

Writes (via _fig_registry):
  viz/descriptive_stats/fig-archive/<ts>__chip_distributions_by_region.png
  viz/descriptive_stats/figures.md (row appended or updated)

Usage:
  python scripts/make_fig_chip_distributions_by_region.py

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_chip_distributions_by_region.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _fig_registry import write as write_fig

LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

DEFAULT_MET_CSV = os.path.join(LLINKAS, "reference/met_data.csv")
DEFAULT_VIZ_DIR = os.path.join(LLINKAS, "viz/descriptive_stats")

REGION_ORDER = ["KQ", "SK"]
REGION_COLORS = {"KQ": "#1976D2", "SK": "#D32F2F"}
REGION_LABELS = {
    "KQ": "Kangerlussuaq (KQ), SE Greenland",
    "SK": "Sermilik (SK), SE Greenland",
}

MONTH_ORDER = list(range(7, 12))  # Jul through Nov, our active window
MONTH_NAMES = {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--met_csv", default=DEFAULT_MET_CSV)
    parser.add_argument("--viz_dir", default=DEFAULT_VIZ_DIR)
    args = parser.parse_args()
    os.makedirs(args.viz_dir, exist_ok=True)

    # 1. Load met data and parse month from date column
    df = pd.read_csv(args.met_csv)
    required = {"chip_stem", "region", "date", "wind_speed_10m", "temp_2m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {args.met_csv}: {missing}")

    df = df[df["region"].isin(REGION_ORDER)].copy()
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
    df = df.dropna(subset=["month"]).copy()
    df["month"] = df["month"].astype(int)

    # 2. Plot 3 panels: month bars, wind histogram, temp histogram
    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    ax_month, ax_wind, ax_temp = axes

    # 3a. Stacked bar of chip count by month, colored by region
    counts = (
        df.groupby(["month", "region"]).size().unstack(fill_value=0)
        .reindex(index=MONTH_ORDER, columns=REGION_ORDER, fill_value=0)
    )
    bottom = np.zeros(len(MONTH_ORDER), dtype=int)
    for region in REGION_ORDER:
        vals = counts[region].values
        ax_month.bar(
            [MONTH_NAMES[m] for m in MONTH_ORDER], vals,
            bottom=bottom, color=REGION_COLORS[region],
            label=f"{REGION_LABELS[region]} (n={vals.sum()})",
            edgecolor="white", linewidth=0.5,
        )
        bottom = bottom + vals
    ax_month.set_xlabel("Acquisition month")
    ax_month.set_ylabel("Chip count")
    ax_month.set_title("a. Chip count by acquisition month",
                       fontsize=11, fontweight="bold", loc="left")
    ax_month.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax_month.grid(True, axis="y", alpha=0.3, linestyle="--")

    # 3b. Wind speed histogram, colored by region (overlapping with alpha)
    wind = df.dropna(subset=["wind_speed_10m"]).copy()
    wind["wind_speed_10m"] = wind["wind_speed_10m"].astype(float)
    wind_bins = np.linspace(0, max(15.0, wind["wind_speed_10m"].max() + 1), 30)
    for region in REGION_ORDER:
        sub = wind[wind["region"] == region]["wind_speed_10m"].values
        if len(sub) == 0:
            continue
        ax_wind.hist(sub, bins=wind_bins, color=REGION_COLORS[region],
                     alpha=0.55, edgecolor="white", linewidth=0.3,
                     label=f"{REGION_LABELS[region]} (n={len(sub)})")
    ax_wind.axvline(15.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                    label=r"15 m s$^{-1}$ confound limit")
    ax_wind.set_xlabel(r"ERA5 10 m wind speed (m s$^{-1}$)")
    ax_wind.set_ylabel("Chip count")
    ax_wind.set_title("b. ERA5 wind speed at chip acquisition time",
                      fontsize=11, fontweight="bold", loc="left")
    ax_wind.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax_wind.grid(True, axis="y", alpha=0.3, linestyle="--")

    # 3c. Air temperature histogram, colored by region
    temp = df.dropna(subset=["temp_2m"]).copy()
    temp["temp_2m"] = temp["temp_2m"].astype(float)
    temp_bins = np.linspace(temp["temp_2m"].min() - 1, temp["temp_2m"].max() + 1, 35)
    for region in REGION_ORDER:
        sub = temp[temp["region"] == region]["temp_2m"].values
        if len(sub) == 0:
            continue
        ax_temp.hist(sub, bins=temp_bins, color=REGION_COLORS[region],
                     alpha=0.55, edgecolor="white", linewidth=0.3,
                     label=f"{REGION_LABELS[region]} (n={len(sub)})")
    ax_temp.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                    label=r"0 $^{\circ}$C freezing line")
    ax_temp.set_xlabel(r"ERA5 2 m air temperature ($^{\circ}$C)")
    ax_temp.set_ylabel("Chip count")
    ax_temp.set_title("c. ERA5 2 m air temperature at chip acquisition time",
                      fontsize=11, fontweight="bold", loc="left")
    ax_temp.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax_temp.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Sentinel-2 chip distributions across SE Greenland (KQ vs SK)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    # 4. Route through fig_registry
    archive = write_fig(
        fig,
        slug="chip_distributions_by_region",
        caption=(
            "Sentinel-2 chip distributions across our two SE Greenland fjords "
            "(KQ + SK). a: chip count by acquisition month, stacked by "
            "region. b: ERA5 10 m wind speed, with the 15 m s^-1 confound "
            "limit dashed. c: ERA5 2 m air temperature, with the 0 C freezing "
            "line dashed. Mirrors Fisser (2025) Fig. 8 + Fig. 9 layout."
        ),
        out_dir=args.viz_dir,
    )
    plt.close(fig)
    print(f"Wrote {archive}")


if __name__ == "__main__":
    main()
