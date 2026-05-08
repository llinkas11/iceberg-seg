"""
make_fig_area_scatter.py: Reference vs predicted iceberg area scatter, one
panel per SZA bin with all six methods overlaid. Solid black 1:1 line and a
shaded +-25 percent band mark the over- and under-estimation regions.
Per-method matching colors tie scatter points to OLS fit lines, and each
panel inset reports n_matched / n_gt (recall fraction) plus MAE on iceberg
root length (m) so the annotation cross-references Fig. 6 and Table 4.

Reads:
  <run>/per_iceberg/eval_per_iceberg.csv         (per-pair rows)
  <run>/per_iceberg/eval_per_iceberg_chips.csv   (per-chip n_gt / n_matched)

Writes (via _fig_registry):
  <run>/per_iceberg/fig-archive/<ts>__area_scatter_by_method.png
  <run>/per_iceberg/figures.md (row appended or updated)
  <run>/per_iceberg/area_scatter_fits.csv (per (method, sza_bin) fit + MAE)

Usage:
  python scripts/make_fig_area_scatter.py \
      --run /mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158

Rsync after edit:
  rsync -av /Users/llinkas/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg-rework/scripts/make_fig_area_scatter.py \
      llinkas@moosehead.bowdoin.edu:/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/scripts/
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from _fig_registry import write as write_fig

DEFAULT_RUN = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework/runs/exp_baseline_v1/20260424_185158"

METHOD_ORDER = ["TR", "OT", "UNet", "UNet_TR", "UNet_OT", "UNet_CRF"]

# Two color families: cool blue-greys for non-learning (TR, OT); warm
# tones for learning. UNet light orange, UNet_TR pink, UNet_OT red,
# UNet_CRF bright purple - mixed warm + purple so all four learning
# methods are distinguishable at small point size.
METHOD_COLORS = {
    "TR":       "#90A4AE",
    "OT":       "#37474F",
    "UNet":     "#FFB74D",
    "UNet_TR":  "#E91E63",
    "UNet_OT":  "#E64A19",
    "UNet_CRF": "#7C4DFF",
}

# Per-method marker shapes harmonized with Fig. 8 / Fig. 9 so the same
# method renders as the same shape across the headline data figures.
METHOD_MARKERS = {
    "TR":       "o",
    "OT":       "s",
    "UNet":     "D",
    "UNet_TR":  "^",
    "UNet_OT":  "v",
    "UNet_CRF": "P",
}

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]
SZA_LABELS = {
    "sza_lt65":  r"SZA $<$ 65$^{\circ}$",
    "sza_65_70": r"SZA 65--70$^{\circ}$",
    "sza_70_75": r"SZA 70--75$^{\circ}$",
    "sza_gt75":  r"SZA $>$ 75$^{\circ}$",
}

# Multiplicative tolerance band around the 1:1 line. Pred above the upper
# bound is overestimation; pred below the lower bound is underestimation.
BAND_FRACTION = 0.25


def fit_linear(x, y):
    """Return slope, intercept, Pearson r, n_used. NaN-tolerant for n<2."""
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    slope, intercept = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)
    return float(slope), float(intercept), float(r), int(len(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()

    # 1. Load per-pair CSV (matched pairs only)
    pairs_path = os.path.join(args.run, "per_iceberg", "eval_per_iceberg.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(pairs_path)
    df = pd.read_csv(pairs_path)
    required = {"method", "sza_bin", "gt_area_m2", "pred_area_m2",
                "abs_rootlen_err_m", "abs_area_err_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {pairs_path}: {missing}")

    # 2. Load chip-level CSV so n_pairs in the inset table is reported as
    # n_matched / n_gt. The denominator is the total reference iceberg count
    # for that (method, SZA bin), matching the recall in Table 6.
    chips_path = os.path.join(args.run, "per_iceberg",
                              "eval_per_iceberg_chips.csv")
    if not os.path.exists(chips_path):
        raise FileNotFoundError(chips_path)
    chips = pd.read_csv(chips_path)
    bin_totals = chips.groupby(["method", "sza_bin"]).agg(
        n_gt_total=("n_gt", "sum"),
        n_matched_total=("n_matched", "sum"),
    ).reset_index()
    totals_lookup = {
        (row["method"], row["sza_bin"]):
            (int(row["n_gt_total"]), int(row["n_matched_total"]))
        for _, row in bin_totals.iterrows()
    }

    # 3. Per (method, sza_bin) compute fit + MAE rootlen + MAE area
    fit_rows = []
    for sza in SZA_ORDER:
        for method in METHOD_ORDER:
            sub = df[(df["method"] == method) & (df["sza_bin"] == sza)]
            slope, intercept, r, n_pairs = fit_linear(
                sub["gt_area_m2"].values, sub["pred_area_m2"].values,
            )
            mae_rl = (float(sub["abs_rootlen_err_m"].mean())
                      if n_pairs > 0 else np.nan)
            mae_area = (float(sub["abs_area_err_m2"].mean())
                        if n_pairs > 0 else np.nan)
            n_gt, n_matched = totals_lookup.get((method, sza), (0, 0))
            match_rate = (n_matched / n_gt) if n_gt else 0.0
            fit_rows.append({
                "sza_bin":         sza,
                "method":          method,
                "n_pairs":         n_pairs,
                "n_gt_total":      n_gt,
                "n_matched_total": n_matched,
                "match_rate":      match_rate,
                "slope":           slope,
                "intercept_m2":    intercept,
                "pearson_r":       r,
                "mae_rootlen_m":   mae_rl,
                "mae_area_m2":     mae_area,
            })

    # 4. Persist fit table next to the figure for cross-checking
    fit_csv = os.path.join(args.run, "per_iceberg",
                           "area_scatter_fits.csv")
    fields = ["sza_bin", "method", "n_pairs", "n_gt_total",
              "n_matched_total", "match_rate", "slope", "intercept_m2",
              "pearson_r", "mae_rootlen_m", "mae_area_m2"]
    with open(fit_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in fit_rows:
            writer.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v)
                for k, v in row.items()
            })
    print(f"Wrote {fit_csv}")

    # 5. Shared axis limits across panels for cross-bin comparability
    upper = float(np.percentile(
        np.concatenate([df["gt_area_m2"].values, df["pred_area_m2"].values]),
        99.0,
    ))
    upper = max(upper, 1e4)
    axis_max = float(np.ceil(upper / 5e4)) * 5e4

    rows_by_sza = {sza: [r for r in fit_rows if r["sza_bin"] == sza]
                   for sza in SZA_ORDER}

    # 6. Plot 2x2 panel grid, one SZA bin per panel
    fig, axes = plt.subplots(2, 2, figsize=(13, 12),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ax, sza in zip(axes_flat, SZA_ORDER):
        # 6a. Shaded +-BAND_FRACTION band around 1:1, with thin dotted bounds
        xs_band = np.linspace(0, axis_max, 200)
        ax.fill_between(xs_band,
                        xs_band * (1 - BAND_FRACTION),
                        xs_band * (1 + BAND_FRACTION),
                        color="#000000", alpha=0.06, zorder=1)
        ax.plot(xs_band, xs_band * (1 + BAND_FRACTION),
                color="#222", linewidth=0.6, linestyle=":", alpha=0.5,
                zorder=2)
        ax.plot(xs_band, xs_band * (1 - BAND_FRACTION),
                color="#222", linewidth=0.6, linestyle=":", alpha=0.5,
                zorder=2)

        # 6b. Solid black 1:1 line
        ax.plot([0, axis_max], [0, axis_max],
                color="black", linewidth=1.3, linestyle="-", alpha=0.9,
                zorder=3)

        # 6c. Per-method scatter + matching-color OLS fit line. Marker shape
        # also varies per method so cross-figure (Fig. 8 / Fig. 9) lookup
        # works on shape as well as color, helping color-blind readers.
        for method in METHOD_ORDER:
            sub = df[(df["method"] == method) & (df["sza_bin"] == sza)]
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            if len(sub) > 0:
                ax.scatter(sub["gt_area_m2"], sub["pred_area_m2"],
                           s=22, alpha=0.45, color=color, marker=marker,
                           edgecolors="none", zorder=4)
            row = next(r for r in fit_rows
                       if r["sza_bin"] == sza and r["method"] == method)
            if np.isfinite(row["slope"]):
                xs2 = np.array([0, axis_max])
                ys2 = row["slope"] * xs2 + row["intercept_m2"]
                ax.plot(xs2, ys2, color=color, linewidth=1.8, alpha=0.95,
                        zorder=5)

        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3, linestyle="--")
        # Force y-tick labels on every panel (sharey would hide them on the
        # right column otherwise). The y-axis text label stays on the left
        # column only; see step 7.
        ax.tick_params(axis="both", labelsize=13, labelleft=True)
        ax.set_title(SZA_LABELS[sza], fontsize=17, fontweight="bold", pad=6)

        # 6d. Inset stat table: method, n_matched / n_gt, MAE root length.
        # Each row is rendered as a separate ax.text so the row text takes
        # the same color as the method's line/points; eye-tracking from a
        # row to its line is direct.
        rows = rows_by_sza[sza]
        # Background plate
        ax.add_patch(mpatches.Rectangle(
            (0.02, 0.62), 0.40, 0.36,
            transform=ax.transAxes,
            facecolor="white", edgecolor="#bbb", linewidth=0.8,
            alpha=0.92, zorder=8,
        ))
        # Header (dark grey, bold)
        ax.text(0.03, 0.96, "method     n_match/n_gt   MAE_rl",
                transform=ax.transAxes, ha="left", va="top",
                family="monospace", fontsize=12, weight="bold",
                color="#222", zorder=9)
        # Per-method rows colored by METHOD_COLORS so the row visually
        # matches its scatter / fit line.
        for i, row in enumerate(rows):
            method = row["method"]
            line = (
                f"{method:<9}  "
                f"{row['n_matched_total']:>4}/{row['n_gt_total']:<4}  "
                f"{row['mae_rootlen_m']:>5.1f} m"
            )
            ax.text(0.03, 0.92 - i * 0.045, line,
                    transform=ax.transAxes, ha="left", va="top",
                    family="monospace", fontsize=12, weight="bold",
                    color=METHOD_COLORS[method], zorder=9)

    # 7. Axis labels. The x label is a single centered figure-level label
    # (fig.supxlabel) instead of one per bottom panel, and the y axis label
    # appears only on the left column. Y-tick numbers stay on every panel
    # (set above via tick_params).
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Predicted iceberg area $A_{method}$ (m$^2$)",
                      fontsize=16)
    fig.supxlabel(r"Reference iceberg area $A_{S2}$ (m$^2$)",
                  fontsize=16, y=0.06)

    # 8. Figure-level legend: 1:1 + band + 6 methods. Method handles use
    # both color and marker shape so the legend lookup matches Fig. 8 /
    # Fig. 9 (same method = same color + shape across the figure set).
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.3, label="1:1"),
        mpatches.Patch(facecolor="#000000", alpha=0.06,
                       label=f"$\\pm${int(BAND_FRACTION * 100)}% band"),
    ]
    for m in METHOD_ORDER:
        handles.append(plt.Line2D(
            [0], [0], color=METHOD_COLORS[m], linewidth=2.2,
            marker=METHOD_MARKERS[m], markersize=8, label=m,
        ))
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=14, frameon=False,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Predicted vs reference iceberg area, per SZA bin",
        fontsize=19, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.965])

    archive = write_fig(
        fig,
        slug="area_scatter_by_method",
        caption=(
            "Per-pair predicted vs reference iceberg area, one panel per "
            "SZA bin, with the 1:1 line solid black and a shaded "
            f"+-{int(BAND_FRACTION * 100)}% band marking over- and "
            "under-estimation regions. Each method overlays a colored "
            "point cloud and matching-color OLS fit line; cool blue-grey "
            "tones mark non-learning methods (TR, OT) and warm orange-red "
            "tones mark learning methods (UNet light orange, UNet_TR "
            "pink, UNet_OT red, UNet_CRF bright purple). Inset table "
            "reports per-method n_matched / n_gt "
            "(matched pairs over total reference icebergs in the panel; "
            "the ratio equals the match rate / recall in Table 6) and "
            "MAE on iceberg root length (m). MAE values match Fig. 6 and "
            "Table 4 entries for the same SZA bin."
        ),
        out_dir=os.path.join(args.run, "per_iceberg"),
    )
    plt.close(fig)
    print(f"Wrote {archive}")

    # 9. Print fit summary to stdout for quick paper text reference
    print("\nPer-(method, sza_bin) fit summary:")
    print(f"{'sza_bin':<12}{'method':<10}{'n_match/n_gt':<14}"
          f"{'match_rate':>11}{'mae_rl_m':>10}{'slope':>9}{'pearson_r':>11}")
    for row in fit_rows:
        ratio = f"{row['n_matched_total']}/{row['n_gt_total']}"
        print(f"{row['sza_bin']:<12}{row['method']:<10}{ratio:<14}"
              f"{row['match_rate']:>11.3f}"
              f"{row['mae_rootlen_m']:>10.2f}"
              f"{row['slope']:>9.3f}"
              f"{row['pearson_r']:>11.3f}")


if __name__ == "__main__":
    main()
