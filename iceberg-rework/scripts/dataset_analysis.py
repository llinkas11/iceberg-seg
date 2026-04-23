"""
dataset_analysis.py — Dataset balance and scene condition analysis

Outputs:
  1. Null vs iceberg chip counts + ratio by split × SZA bin
  2. Iceberg root-length stats (sqrt(area_m2)) by SZA bin
  3. Scene-level wind speed / temperature from Open-Meteo ERA5 reanalysis
     Flags scenes passing: wind_speed_10m <= 15 m/s  AND  temp_2m > 0°C

Usage:
  python dataset_analysis.py
  python dataset_analysis.py --out_csv analysis_output.csv --no_met
"""

import pickle
import argparse
import re
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Paths — works on both Mac (repo root) and moosehead
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# moosehead research volume (if it exists, prefer it for pkl + scene catalogue)
_MOOSE_ROOT = Path("/mnt/research/v.gomezgilyaspik/students/smishra/S2-iceberg-areas")
if _MOOSE_ROOT.exists():
    PKL_DIR   = _MOOSE_ROOT / "train_validate_test_v2" / "train_validate_test"
    SPLIT_LOG = _MOOSE_ROOT / "train_validate_test_v2" / "split_log.csv"
    SCENE_CAT = _MOOSE_ROOT / "scene_catalogue.csv"
else:
    PKL_DIR   = ROOT / "train_validate_test_v2" / "train_validate_test"
    SPLIT_LOG = ROOT / "train_validate_test_v2" / "split_log.csv"
    SCENE_CAT = ROOT / "scene_catalogue.csv"

PIXEL_AREA_M2 = 100.0   # 10m × 10m

# Approx centre coordinates for each region (used for ERA5 met query)
REGION_COORDS = {
    "KQ": (68.60, -32.50),   # Kangerlussuaq Fjord, East Greenland
    "SK": (65.70, -38.00),   # Sermilik Fjord, SE Greenland
}

# Met thresholds
WIND_MAX_MS  = 15.0   # m/s
TEMP_MIN_C   =  0.0   # °C


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load masks + split log → per-chip DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_masks():
    splits = [
        ("train", "X_train.pkl",      "Y_train.pkl"),
        ("val",   "X_validation.pkl", "Y_validation.pkl"),
        ("test",  "x_test.pkl",       "y_test.pkl"),
    ]
    rows = []
    for split_name, _, yfile in splits:
        with open(PKL_DIR / yfile, "rb") as f:
            y = np.array(pickle.load(f))   # (N, 1, 256, 256) int64
        y = y[:, 0, :, :]                  # (N, 256, 256)
        for i in range(len(y)):
            mask = y[i]
            n_iceberg = int((mask == 1).sum())
            n_shadow  = int((mask == 2).sum())
            n_ocean   = int((mask == 0).sum())
            n_total   = mask.size
            rows.append({
                "split":         split_name,
                "pkl_position":  i,
                "n_iceberg_px":  n_iceberg,
                "n_shadow_px":   n_shadow,
                "n_ocean_px":    n_ocean,
                "has_iceberg":   n_iceberg > 0,
                "iceberg_frac":  n_iceberg / n_total,
                "iceberg_area_m2": n_iceberg * PIXEL_AREA_M2,
                "root_length_m": np.sqrt(n_iceberg * PIXEL_AREA_M2) if n_iceberg > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def merge_split_log(chips_df):
    log = pd.read_csv(SPLIT_LOG)
    # split_log 'index' is the global position in the combined 984-chip list —
    # NOT the within-split pkl slot.  The pkl files are packed in the same row
    # order as split_log rows within each split group, so we use cumulative row
    # rank within each split as the pkl_position key.
    log["pkl_position"] = log.groupby("split").cumcount()
    merged = chips_df.merge(log[["split", "pkl_position", "stem", "sza_bin"]],
                            on=["split", "pkl_position"], how="left")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Null / iceberg summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_null_iceberg_table(df):
    print("\n" + "="*70)
    print("NULL vs ICEBERG CHIPS by SPLIT × SZA BIN")
    print("="*70)

    summary = (
        df.groupby(["split", "sza_bin"])
        .agg(
            total=("has_iceberg", "count"),
            iceberg_chips=("has_iceberg", "sum"),
        )
        .assign(null_chips=lambda x: x["total"] - x["iceberg_chips"])
        .assign(null_to_iceberg_ratio=lambda x: (x["null_chips"] / x["iceberg_chips"]).round(2))
        .assign(iceberg_pct=lambda x: (x["iceberg_chips"] / x["total"] * 100).round(1))
    )

    print(summary.to_string())

    print("\n--- COMBINED (all splits) ---")
    combined = (
        df.groupby("sza_bin")
        .agg(
            total=("has_iceberg", "count"),
            iceberg_chips=("has_iceberg", "sum"),
        )
        .assign(null_chips=lambda x: x["total"] - x["iceberg_chips"])
        .assign(null_to_iceberg_ratio=lambda x: (x["null_chips"] / x["iceberg_chips"]).round(2))
        .assign(iceberg_pct=lambda x: (x["iceberg_chips"] / x["total"] * 100).round(1))
    )
    print(combined.to_string())
    return summary, combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. Root-length stats
# ─────────────────────────────────────────────────────────────────────────────

def print_root_length_stats(df):
    print("\n" + "="*70)
    print("ICEBERG ROOT LENGTH (sqrt of iceberg area) by SZA BIN")
    print("  root_length_m = sqrt(n_iceberg_pixels × 100 m²)")
    print("  = characteristic linear size of all labeled iceberg pixels per chip")
    print("="*70)

    pos = df[df["has_iceberg"]].copy()

    stats = (
        pos.groupby("sza_bin")["root_length_m"]
        .agg(
            count="count",
            mean=lambda x: round(x.mean(), 1),
            std=lambda x: round(x.std(), 1),
            p25=lambda x: round(x.quantile(0.25), 1),
            median=lambda x: round(x.median(), 1),
            p75=lambda x: round(x.quantile(0.75), 1),
            max=lambda x: round(x.max(), 1),
        )
    )
    print(stats.to_string())

    print("\n--- ALL SZA BINS COMBINED ---")
    r = pos["root_length_m"]
    print(f"  n chips (iceberg>0) : {len(r)}")
    print(f"  mean   : {r.mean():.1f} m")
    print(f"  std    : {r.std():.1f} m")
    print(f"  p25    : {r.quantile(0.25):.1f} m")
    print(f"  median : {r.median():.1f} m")
    print(f"  p75    : {r.quantile(0.75):.1f} m")
    print(f"  max    : {r.max():.1f} m")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Met data from Open-Meteo ERA5 (free, no API key)
# ─────────────────────────────────────────────────────────────────────────────

def parse_scene_date(stem):
    """Extract acquisition datetime from S2 scene stem.
    Format: S2X_MSIL1C_YYYYMMDDTHHMMSS_...
    Returns (date_str 'YYYY-MM-DD', hour_utc int) or (None, None) for Fisser."""
    m = re.match(r"S2[AB]_MSIL1C_(\d{8})T(\d{6})_", stem)
    if m:
        date_str = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
        hour_utc = int(m.group(2)[:2])
        return date_str, hour_utc
    return None, None


def fetch_met_openmeteo(lat, lon, date_str, hour_utc, retries=3):
    """Fetch ERA5 hourly wind speed and temperature for one date/location.
    Returns (wind_speed_ms, temp_c) at the acquisition hour, or (None, None)."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly":     "temperature_2m,wind_speed_10m",
        "timezone":   "UTC",
        "wind_speed_unit": "ms",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            hours = data["hourly"]["time"]           # list of "YYYY-MM-DDTHH:00"
            temps = data["hourly"]["temperature_2m"]
            winds = data["hourly"]["wind_speed_10m"]
            # pick closest hour
            target = f"{date_str}T{hour_utc:02d}:00"
            if target in hours:
                idx = hours.index(target)
            else:
                idx = min(range(len(hours)),
                          key=lambda i: abs(int(hours[i][11:13]) - hour_utc))
            return winds[idx], temps[idx]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None, None


def infer_region(stem, scene_cat):
    """Try to infer region (KQ/SK) from stem via scene_catalogue."""
    if scene_cat is not None:
        scene_stem = stem.split(".SAFE")[0] if ".SAFE" in stem else stem
        row = scene_cat[scene_cat["name"].str.startswith(scene_stem)]
        if not row.empty:
            return row.iloc[0]["region"]
    # Fisser chips or unknown → assume KQ (sza_lt65)
    return "KQ"


def build_met_table(df, scene_cat, fetch=True):
    print("\n" + "="*70)
    print("METEOROLOGICAL CONDITIONS (Open-Meteo ERA5)")
    print(f"  Thresholds: wind_speed_10m <= {WIND_MAX_MS} m/s  AND  temp_2m > {TEMP_MIN_C} °C")
    print("="*70)

    # Unique scene stems (strip row/col suffix if present)
    # split_log stems are already scene-level (no _rXXXX_cXXXX)
    unique_stems = df[["stem", "sza_bin"]].dropna(subset=["stem"]).drop_duplicates("stem")

    if not fetch:
        print("  [--no_met flag set — skipping API fetch]")
        return None

    records = []
    n = len(unique_stems)
    print(f"  Fetching met data for {n} unique scene stems ...\n")

    for i, (_, row) in enumerate(unique_stems.iterrows()):
        stem    = row["stem"]
        sza_bin = row["sza_bin"]
        date_str, hour_utc = parse_scene_date(stem)

        if date_str is None:
            # Fisser chip — no acquisition time in stem name
            records.append({"stem": stem, "sza_bin": sza_bin,
                             "date": None, "hour_utc": None,
                             "wind_ms": None, "temp_c": None,
                             "pass_wind": None, "pass_temp": None, "pass_both": None})
            continue

        region  = infer_region(stem, scene_cat)
        lat, lon = REGION_COORDS[region]
        wind, temp = fetch_met_openmeteo(lat, lon, date_str, hour_utc)

        pass_wind = (wind <= WIND_MAX_MS)  if wind is not None else None
        pass_temp = (temp > TEMP_MIN_C)    if temp is not None else None
        pass_both = (pass_wind and pass_temp) if (pass_wind is not None) else None

        records.append({
            "stem":      stem,
            "region":    region,
            "sza_bin":   sza_bin,
            "date":      date_str,
            "hour_utc":  hour_utc,
            "wind_ms":   round(wind, 2)  if wind is not None else None,
            "temp_c":    round(temp, 2)  if temp is not None else None,
            "pass_wind": pass_wind,
            "pass_temp": pass_temp,
            "pass_both": pass_both,
        })

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}] {stem[:55]}  wind={wind}  temp={temp}")
        time.sleep(0.15)   # polite rate limiting

    met_df = pd.DataFrame(records)

    # Summary by SZA bin
    met_with_data = met_df.dropna(subset=["pass_both"])
    print("\n--- Scenes passing wind<=15 m/s AND temp>0°C by SZA bin ---")
    summary = (
        met_with_data.groupby("sza_bin")
        .agg(
            total_scenes=("stem", "count"),
            pass_wind=("pass_wind", "sum"),
            pass_temp=("pass_temp", "sum"),
            pass_both=("pass_both", "sum"),
        )
        .assign(pass_pct=lambda x: (x["pass_both"] / x["total_scenes"] * 100).round(1))
    )
    print(summary.to_string())

    # Count how many chips (not scenes) pass thresholds
    chips_met = df.merge(met_df[["stem","pass_both","wind_ms","temp_c"]],
                         on="stem", how="left")
    print("\n--- Chips in passing scenes by SZA bin ---")
    pass_chips = (
        chips_met[chips_met["pass_both"] == True]
        .groupby("sza_bin")
        .agg(
            total_chips=("has_iceberg", "count"),
            iceberg_chips=("has_iceberg", "sum"),
        )
        .assign(null_chips=lambda x: x["total_chips"] - x["iceberg_chips"])
    )
    print(pass_chips.to_string())

    return met_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv",  default="dataset_analysis_output.csv",
                        help="Save per-chip DataFrame to CSV")
    parser.add_argument("--no_met",   action="store_true",
                        help="Skip Open-Meteo API calls")
    args = parser.parse_args()

    print("Loading masks ...")
    df = load_masks()
    df = merge_split_log(df)
    print(f"  Loaded {len(df)} chips  ({df['sza_bin'].isna().sum()} with missing sza_bin)")

    # ── Section 1: null / iceberg balance ──
    null_summary, combined_summary = print_null_iceberg_table(df)

    # ── Section 2: root-length stats ──
    rl_stats = print_root_length_stats(df)

    # ── Section 3: met data ──
    scene_cat = None
    if SCENE_CAT.exists():
        scene_cat = pd.read_csv(SCENE_CAT)

    met_df = build_met_table(df, scene_cat, fetch=not args.no_met)

    # ── Save per-chip CSV ──
    out = Path(args.out_csv)
    df.to_csv(out, index=False)
    print(f"\nPer-chip data saved → {out}")

    if met_df is not None:
        met_out = out.with_name(out.stem + "_met.csv")
        met_df.to_csv(met_out, index=False)
        print(f"Met data saved      → {met_out}")


if __name__ == "__main__":
    main()
