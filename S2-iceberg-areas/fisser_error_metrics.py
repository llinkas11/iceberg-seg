"""
fisser_error_metrics.py

Compute Fisser-style iceberg area error metrics (RE, MAE, IoU) per matched
iceberg pair for every (method, region, sza_bin, chip) combination, then
aggregate across (a) iceberg size and (b) solar zenith angle.

Reference set: visually delineated polygons exported by
  iceberg-labeler/scripts/export_reference_gpkg.py
at {pred_root}/{REGION}/{sza_bin}/reference/gpkgs/{chip_stem}_reference.gpkg.

Method predictions (Sentinel-2):
  unet       : {pred_root}/{REGION}/{sza_bin}/unet/gpkgs/{chip_stem}_icebergs.gpkg
  threshold  : {pred_root}/{REGION}/{sza_bin}/threshold/all_icebergs_threshold.gpkg
  otsu       : {pred_root}/{REGION}/{sza_bin}/otsu/all_icebergs_otsu.gpkg
  densecrf   : {pred_root}/{REGION}/{sza_bin}/densecrf/all_icebergs_densecrf.gpkg
Methods whose GPKGs are missing are tolerated with a warning.

Pairing: per chip, IoU matrix between reference and prediction polygons.
Hungarian assignment (scipy.optimize.linear_sum_assignment) on cost
(1 - IoU), keep pairs with IoU >= iou_threshold (default 0.3).

Metrics on matched pairs (per pair):
  RE_pct           = 100 * (A_pred - A_ref) / A_ref              [Fisser Eq. 2]
  AE_area_m2       = |A_pred - A_ref|
  AE_rootlen_m     = |sqrt(A_pred) - sqrt(A_ref)|
  iou              = intersection / union

Aggregations:
  per_pair.csv        one row per matched pair
  re_by_size.csv      RE + MAE stats stratified by ref root-length bucket
  re_by_sza.csv       per-degree median RE, 1-deg linear interp (Eq. 3) +
                      5-deg centered running mean (Eq. 4). No SRE (Eq. 5
                      intentionally not applied, see methods_draft.md).
  detection_stats.csv n_ref, n_pred, n_matched, match_rate, precision

Usage:
  python fisser_error_metrics.py \\
      --pred_root /mnt/research/.../area_comparison \\
      --sza_csv   /mnt/research/.../area_comparison/chip_sza.csv \\
      --out_root  /mnt/research/.../figures/fisser

Rsync:
  rsync -av /Users/smishra/S2-iceberg-areas/fisser_error_metrics.py \\
      smishra@moosehead.bowdoin.edu:~/S2-iceberg-areas/
"""

import os
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_PRED_ROOT = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/area_comparison"
DEFAULT_SZA_CSV   = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/area_comparison/chip_sza.csv"
DEFAULT_OUT_ROOT  = "/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas/figures/fisser"

SZA_ORDER = ["sza_lt65", "sza_65_70", "sza_70_75", "sza_gt75"]

METHOD_PATHS = {
    # per-chip gpkg directory (preferred)
    "unet":      {"per_chip_dir": "unet/gpkgs",      "suffix": "_icebergs.gpkg",
                  "all_file":     "unet/all_icebergs.gpkg"},
    # per-bin combined gpkg (filtered on source_file column)
    "threshold": {"per_chip_dir": "threshold/gpkgs", "suffix": "_icebergs.gpkg",
                  "all_file":     "threshold/all_icebergs_threshold.gpkg"},
    "otsu":      {"per_chip_dir": "otsu/gpkgs",      "suffix": "_icebergs.gpkg",
                  "all_file":     "otsu/all_icebergs_otsu.gpkg"},
    "densecrf":  {"per_chip_dir": "densecrf/gpkgs",  "suffix": "_icebergs.gpkg",
                  "all_file":     "densecrf/all_icebergs_densecrf.gpkg"},
}

SIZE_BUCKETS = [
    (40,   80,    "40-80"),
    (80,   160,   "80-160"),
    (160,  320,   "160-320"),
    (320,  640,   "320-640"),
    (640,  np.inf, ">=640"),
]


# 1. Loading helpers ---------------------------------------------------------

def load_reference_chip(pred_root, region, sza_bin, chip_stem):
    """Read the per-chip reference GPKG; return GeoDataFrame or None."""
    path = Path(pred_root) / region.upper() / sza_bin / "reference" / "gpkgs" / \
           f"{chip_stem}_reference.gpkg"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf if len(gdf) else None


def load_method_chip(pred_root, region, sza_bin, chip_stem, method,
                     all_cache):
    """
    Load a method's polygons for ONE chip. First try the per-chip GPKG;
    fall back to the per-bin all_icebergs file (cached in all_cache) and
    filter on source_file == chip_stem + '.tif'.
    Returns a GeoDataFrame (possibly empty) or None if method has no data.
    """
    paths = METHOD_PATHS[method]
    per_chip = Path(pred_root) / region.upper() / sza_bin / paths["per_chip_dir"] / \
               f"{chip_stem}{paths['suffix']}"
    if per_chip.exists():
        gdf = gpd.read_file(per_chip)
        gdf = gdf[gdf.get("class_name", "iceberg").fillna("iceberg") == "iceberg"]
        return gdf

    all_path = Path(pred_root) / region.upper() / sza_bin / paths["all_file"]
    if not all_path.exists():
        return None
    key = str(all_path)
    if key not in all_cache:
        try:
            all_cache[key] = gpd.read_file(all_path)
        except Exception as e:
            print(f"  [warn] could not read {all_path}: {e}")
            all_cache[key] = None
    full = all_cache[key]
    if full is None:
        return None
    if "class_name" in full.columns:
        full = full[full["class_name"] == "iceberg"]
    # source_file column was written by predict_tifs.py as basename(tif).
    # Chips are named <stem>.tif; match either stem or full basename.
    if "source_file" not in full.columns:
        return full
    src = full["source_file"].astype(str)
    mask = (src == f"{chip_stem}.tif") | (src == chip_stem)
    return full[mask]


# 2. Geometry helpers --------------------------------------------------------

def polygon_area_m2(geom):
    """Shapely .area in UTM CRS == m^2."""
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area)


def iou(a, b):
    """IoU between two Shapely geometries."""
    if a is None or b is None or a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter == 0.0:
        return 0.0
    union = a.union(b).area
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(ref_geoms, pred_geoms):
    """Dense IoU matrix of shape (len(ref), len(pred)). Uses envelope prefilter."""
    n, m = len(ref_geoms), len(pred_geoms)
    M = np.zeros((n, m), dtype=float)
    if n == 0 or m == 0:
        return M
    ref_bounds  = [g.bounds for g in ref_geoms]
    pred_bounds = [g.bounds for g in pred_geoms]
    for i, (rminx, rminy, rmaxx, rmaxy) in enumerate(ref_bounds):
        for j, (pminx, pminy, pmaxx, pmaxy) in enumerate(pred_bounds):
            if rmaxx < pminx or pmaxx < rminx or rmaxy < pminy or pmaxy < rminy:
                continue                                  # disjoint envelopes
            M[i, j] = iou(ref_geoms[i], pred_geoms[j])
    return M


# 3. Matching ----------------------------------------------------------------

def match_pairs(ref_gdf, pred_gdf, iou_threshold, ambig_iou=0.1):
    """
    Hungarian match between reference and prediction polygons within a chip.

    Returns:
      pairs         : list of (ref_idx, pred_idx, iou) for accepted matches
      ambiguous     : list of (ref_idx, [pred_idx,...]) N:M groups dropped
      n_ref, n_pred : counts (for detection stats)
    """
    ref_geoms  = list(ref_gdf.geometry.values)  if ref_gdf  is not None else []
    pred_geoms = list(pred_gdf.geometry.values) if pred_gdf is not None else []
    n_ref, n_pred = len(ref_geoms), len(pred_geoms)
    if n_ref == 0 or n_pred == 0:
        return [], [], n_ref, n_pred

    M = iou_matrix(ref_geoms, pred_geoms)

    # 3a. Drop ambiguous N:M groups (a ref with >=2 preds over ambig_iou).
    ambiguous = []
    ambiguous_ref_ids = set()
    for i in range(n_ref):
        overlaps = np.where(M[i] >= ambig_iou)[0]
        if len(overlaps) >= 2:
            ambiguous.append((i, overlaps.tolist()))
            ambiguous_ref_ids.add(i)

    # 3b. Hungarian on square-padded cost matrix (1 - IoU). Pad with high cost.
    k = max(n_ref, n_pred)
    cost = np.ones((k, k), dtype=float)
    cost[:n_ref, :n_pred] = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)

    # 3c. Keep assignments that are real (not padding), above threshold,
    # and not flagged ambiguous.
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r >= n_ref or c >= n_pred:
            continue
        if r in ambiguous_ref_ids:
            continue
        iou_val = M[r, c]
        if iou_val >= iou_threshold:
            pairs.append((int(r), int(c), float(iou_val)))
    return pairs, ambiguous, n_ref, n_pred


# 4. Per-pair record ---------------------------------------------------------

def build_pair_record(method, region, sza_bin, chip_stem, sza_deg,
                      ref_row, pred_row, iou_val, tags):
    a_ref  = float(ref_row.get("area_m2", ref_row.geometry.area))
    a_pred = float(pred_row.get("area_m2", pred_row.geometry.area))
    re_pct = 100.0 * (a_pred - a_ref) / a_ref if a_ref > 0 else np.nan
    ae_area = abs(a_pred - a_ref)
    ae_root = abs(math.sqrt(max(a_pred, 0)) - math.sqrt(max(a_ref, 0)))
    return {
        "method":              method,
        "region":              region,
        "sza_bin":             sza_bin,
        "chip_stem":           chip_stem,
        "sza_deg":             sza_deg,
        "ref_id":              ref_row.get("iceberg_id"),
        "pred_id":             pred_row.get("iceberg_id",
                                            pred_row.get("id", None)),
        "A_ref_m2":            round(a_ref, 4),
        "A_pred_m2":           round(a_pred, 4),
        "RE_pct":              round(re_pct, 4) if np.isfinite(re_pct) else np.nan,
        "AE_area_m2":          round(ae_area, 4),
        "AE_rootlen_m":        round(ae_root, 4),
        "iou":                 round(iou_val, 4),
        "ref_root_length_m":   round(math.sqrt(max(a_ref,  0)), 4),
        "pred_root_length_m":  round(math.sqrt(max(a_pred, 0)), 4),
        "tags":                tags,
    }


# 5. Aggregations ------------------------------------------------------------

def aggregate_by_size(per_pair_df):
    """RE/MAE stats per method x region x root-length bucket."""
    rows = []
    for method, mdf in per_pair_df.groupby("method"):
        for region, rdf in mdf.groupby("region"):
            for lo, hi, label in SIZE_BUCKETS:
                mask = (rdf["ref_root_length_m"] >= lo) & (rdf["ref_root_length_m"] < hi)
                sub  = rdf[mask]
                if len(sub) == 0:
                    continue
                rows.append({
                    "method":         method,
                    "region":         region,
                    "bucket_label":   label,
                    "bucket_lo_m":    lo,
                    "bucket_hi_m":    hi if np.isfinite(hi) else None,
                    "n_pairs":        len(sub),
                    "RE_mean":        round(float(sub["RE_pct"].mean()), 3),
                    "RE_median":      round(float(sub["RE_pct"].median()), 3),
                    "RE_p25":         round(float(sub["RE_pct"].quantile(0.25)), 3),
                    "RE_p75":         round(float(sub["RE_pct"].quantile(0.75)), 3),
                    "MAE_area_m2":    round(float(sub["AE_area_m2"].mean()), 2),
                    "MAE_rootlen_m":  round(float(sub["AE_rootlen_m"].mean()), 3),
                    "mean_iou":       round(float(sub["iou"].mean()), 4),
                })
    return pd.DataFrame(rows)


def interp_fill(series_deg_to_val):
    """
    Given {deg: value} over integer degrees, fill missing degrees in the
    observed range by linear interpolation (Fisser Eq. 3). Degrees outside
    [min_observed, max_observed] are left NaN (no extrapolation).
    """
    if not series_deg_to_val:
        return {}
    degs = sorted(series_deg_to_val.keys())
    lo, hi = degs[0], degs[-1]
    out = {}
    for d in range(lo, hi + 1):
        if d in series_deg_to_val:
            out[d] = series_deg_to_val[d]
            continue
        # find neighbors a (lower) and b (higher)
        a = max([x for x in degs if x < d], default=None)
        b = min([x for x in degs if x > d], default=None)
        if a is None or b is None:
            out[d] = np.nan
            continue
        va, vb = series_deg_to_val[a], series_deg_to_val[b]
        out[d] = va + (d - a) * (vb - va) / (b - a)
    return out


def running_mean_5deg(deg_to_val):
    """
    5-degree centered running mean (Fisser Eq. 4). At the lower (d0, d0+1)
    and upper (dN-1, dN) edges the window has 3 or 4 samples, matching
    Fisser's handling of endpoints.
    """
    if not deg_to_val:
        return {}
    degs = sorted(deg_to_val.keys())
    lo, hi = degs[0], degs[-1]
    vals = np.array([deg_to_val[d] for d in degs], dtype=float)
    out = {}
    for i, d in enumerate(degs):
        a = max(0, i - 2); b = min(len(degs), i + 3)
        window = vals[a:b]
        window = window[np.isfinite(window)]
        out[d] = float(window.mean()) if len(window) else np.nan
    return out


def aggregate_by_sza(per_pair_df):
    """
    Per method: bin per-pair RE by integer SZA degree, take median (+ p25/p75),
    fill gaps via linear interp, then 5-deg running mean. Emit one row per
    (method, sza_deg).
    """
    rows = []
    for method, mdf in per_pair_df.groupby("method"):
        df = mdf.dropna(subset=["sza_deg", "RE_pct"]).copy()
        if len(df) == 0:
            continue
        df["sza_int"] = df["sza_deg"].round().astype(int)
        by_deg = df.groupby("sza_int")
        med_raw = by_deg["RE_pct"].median().to_dict()
        p25_raw = by_deg["RE_pct"].quantile(0.25).to_dict()
        p75_raw = by_deg["RE_pct"].quantile(0.75).to_dict()
        n_at_deg = by_deg.size().to_dict()

        med_interp = interp_fill(med_raw)
        p25_interp = interp_fill(p25_raw)
        p75_interp = interp_fill(p75_raw)

        med_smooth = running_mean_5deg(med_interp)
        p25_smooth = running_mean_5deg(p25_interp)
        p75_smooth = running_mean_5deg(p75_interp)

        for d in sorted(med_interp.keys()):
            rows.append({
                "method":             method,
                "sza_deg":            int(d),
                "RE_median_raw":      round(med_raw[d], 4) if d in med_raw else np.nan,
                "RE_median_interp":   round(med_interp[d], 4) if np.isfinite(med_interp[d]) else np.nan,
                "RE_median_smooth5":  round(med_smooth[d], 4) if np.isfinite(med_smooth[d]) else np.nan,
                "RE_p25_smooth5":     round(p25_smooth[d], 4) if np.isfinite(p25_smooth[d]) else np.nan,
                "RE_p75_smooth5":     round(p75_smooth[d], 4) if np.isfinite(p75_smooth[d]) else np.nan,
                "n_pairs_at_sza":     int(n_at_deg.get(d, 0)),
            })
    return pd.DataFrame(rows)


# 6. Main --------------------------------------------------------------------

def discover_chips(pred_root, method):
    """
    Enumerate (region, sza_bin, chip_stem) triples present in area_comparison.
    Uses the UNION of reference/ gpkgs and the given method's outputs, so
    every region+bin+chip that has EITHER a reference or a prediction shows
    up (we still require both for RE).
    """
    root = Path(pred_root)
    triples = set()
    for region_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        region = region_dir.name.lower()
        for sza_dir in sorted([d for d in region_dir.iterdir() if d.is_dir()]):
            sza_bin = sza_dir.name
            if sza_bin not in SZA_ORDER:
                continue
            ref_dir = sza_dir / "reference" / "gpkgs"
            if ref_dir.is_dir():
                for f in ref_dir.glob("*_reference.gpkg"):
                    triples.add((region, sza_bin, f.stem.replace("_reference", "")))
            paths = METHOD_PATHS[method]
            mdir = sza_dir / paths["per_chip_dir"]
            if mdir.is_dir():
                for f in mdir.glob(f"*{paths['suffix']}"):
                    triples.add((region, sza_bin, f.stem.replace(paths["suffix"][:-5], "")))
    return sorted(triples)


def main():
    parser = argparse.ArgumentParser(description="Fisser-style per-pair RE/MAE/IoU.")
    parser.add_argument("--pred_root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--sza_csv",   default=DEFAULT_SZA_CSV)
    parser.add_argument("--out_root",  default=DEFAULT_OUT_ROOT)
    parser.add_argument("--methods",   nargs="+",
                        default=["unet", "threshold", "otsu", "densecrf"])
    parser.add_argument("--region",    default=None, help="kq | sk")
    parser.add_argument("--sza_bin",   default=None)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 6a. Load SZA lookup.
    sza_lookup = {}
    if os.path.exists(args.sza_csv):
        sza_df = pd.read_csv(args.sza_csv)
        for _, r in sza_df.iterrows():
            sza_lookup[(str(r["region"]).lower(), str(r["chip_stem"]))] = r["sza_deg"]
    else:
        print(f"[warn] SZA CSV not found at {args.sza_csv}; "
              "RE-by-SZA will use the bin midpoint.")

    bin_midpoint = {
        "sza_lt65": 60.0, "sza_65_70": 67.5, "sza_70_75": 72.5, "sza_gt75": 78.0,
    }

    pair_rows         = []
    detection_rows    = []
    ambiguous_rows    = []
    all_cache         = {}

    # 6b. Iterate methods x chips.
    for method in args.methods:
        if method not in METHOD_PATHS:
            print(f"[skip] unknown method {method}")
            continue
        triples = discover_chips(args.pred_root, method)
        if args.region:
            triples = [t for t in triples if t[0] == args.region.lower()]
        if args.sza_bin:
            triples = [t for t in triples if t[1] == args.sza_bin]
        print(f"\n=== {method}: {len(triples)} (region, sza_bin, chip) triples ===")
        if not triples:
            continue

        per_bin_counts = {}
        for region, sza_bin, chip_stem in triples:
            ref_gdf  = load_reference_chip(args.pred_root, region, sza_bin, chip_stem)
            pred_gdf = load_method_chip(args.pred_root, region, sza_bin, chip_stem,
                                        method, all_cache)

            n_ref  = 0 if ref_gdf  is None else len(ref_gdf)
            n_pred = 0 if pred_gdf is None else len(pred_gdf)

            pairs, ambig, _, _ = match_pairs(ref_gdf, pred_gdf, args.iou_threshold)

            # 6c. SZA for this chip.
            sza_deg = sza_lookup.get((region, chip_stem), bin_midpoint.get(sza_bin, np.nan))

            # 6d. Tag string from reference (identical for all ref polys).
            tags = ""
            if ref_gdf is not None and "tags" in ref_gdf.columns and len(ref_gdf):
                tags = str(ref_gdf.iloc[0]["tags"] or "")

            # 6e. Emit pair records.
            for ref_i, pred_j, iou_val in pairs:
                pair_rows.append(build_pair_record(
                    method, region, sza_bin, chip_stem, sza_deg,
                    ref_gdf.iloc[ref_i], pred_gdf.iloc[pred_j], iou_val, tags,
                ))

            for ref_i, pred_js in ambig:
                ambiguous_rows.append({
                    "method": method, "region": region, "sza_bin": sza_bin,
                    "chip_stem": chip_stem, "ref_idx": ref_i,
                    "pred_idxs": ",".join(str(j) for j in pred_js),
                })

            key = (method, region, sza_bin)
            c = per_bin_counts.setdefault(key,
                {"n_ref": 0, "n_pred": 0, "n_matched": 0, "iou_sum": 0.0})
            c["n_ref"]     += n_ref
            c["n_pred"]    += n_pred
            c["n_matched"] += len(pairs)
            c["iou_sum"]   += sum(p[2] for p in pairs)

        # 6f. Detection stats for this method.
        for (m, region, sza_bin), c in per_bin_counts.items():
            detection_rows.append({
                "method":           m,
                "region":           region,
                "sza_bin":          sza_bin,
                "n_ref":            c["n_ref"],
                "n_pred":           c["n_pred"],
                "n_matched":        c["n_matched"],
                "match_rate":       round(c["n_matched"] / c["n_ref"],  4) if c["n_ref"]  else np.nan,
                "precision":        round(c["n_matched"] / c["n_pred"], 4) if c["n_pred"] else np.nan,
                "mean_iou_matched": round(c["iou_sum"]   / c["n_matched"], 4) if c["n_matched"] else np.nan,
            })

    # 6g. Write outputs.
    pair_df = pd.DataFrame(pair_rows)
    det_df  = pd.DataFrame(detection_rows)
    amb_df  = pd.DataFrame(ambiguous_rows)

    pair_path = out_root / "per_pair.csv"
    det_path  = out_root / "detection_stats.csv"
    amb_path  = out_root / "ambiguous_groups.csv"
    pair_df.to_csv(pair_path, index=False)
    det_df.to_csv(det_path, index=False)
    amb_df.to_csv(amb_path, index=False)
    print(f"\nper_pair.csv          : {len(pair_df)} rows -> {pair_path}")
    print(f"detection_stats.csv   : {len(det_df)} rows -> {det_path}")
    print(f"ambiguous_groups.csv  : {len(amb_df)} rows -> {amb_path}")

    if len(pair_df) == 0:
        print("\n[warn] No matched pairs. Size and SZA aggregates skipped.")
        return

    size_df = aggregate_by_size(pair_df)
    size_path = out_root / "re_by_size.csv"
    size_df.to_csv(size_path, index=False)
    print(f"re_by_size.csv        : {len(size_df)} rows -> {size_path}")

    sza_df = aggregate_by_sza(pair_df)
    sza_path = out_root / "re_by_sza.csv"
    sza_df.to_csv(sza_path, index=False)
    print(f"re_by_sza.csv         : {len(sza_df)} rows -> {sza_path}")

    # 6h. Small-sample flag per (method, region, sza_bin).
    print("\nBins with n_pairs < 50 (medians unreliable):")
    for _, r in det_df.iterrows():
        if (r["n_matched"] or 0) < 50:
            print(f"  {r['method']:10s} {r['region']} {r['sza_bin']:10s} "
                  f"n_matched={r['n_matched']}")


if __name__ == "__main__":
    main()
