"""
fetch_met_data.py — Fetch wind speed and temperature for all chips via Open-Meteo ERA5.

Collects 10 m wind speed and 2 m air temperature at the hour closest to each
Sentinel-2 acquisition. Chips are flagged against thresholds:
  - wind_speed_10m <= 15 m/s
  - temp_2m > 0 C

Uses the Open-Meteo ERA5 archive API (free, no key needed). A CARRA-based
version would require Copernicus CDS API credentials.

Usage:
  python scripts/fetch_met_data.py
  python scripts/fetch_met_data.py --no_fetch   # use cached met_data.csv
"""

import argparse
import csv
import os
import re
import time

import numpy as np
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
SMISHRA = "/mnt/research/v.gomezgilyaspik/students/smishra/rework"
LLINKAS = "/mnt/research/v.gomezgilyaspik/students/llinkas/iceberg-rework"

SPLIT_LOG = os.path.join(SMISHRA, "data/split_log.csv")
PROVENANCE_CSV = os.path.join(LLINKAS, "reference/fisser_provenance_audit.csv")

# Region coordinates for ERA5 query
REGION_COORDS = {
    "KQ": (68.60, -32.50),
    "SK": (65.70, -38.00),
}

# Thresholds
WIND_MAX = 15.0   # m/s
TEMP_MIN = 0.0    # C

# S2 filename date parser
S2_DATE_RE = re.compile(r"S2[AB]_MSIL1C_(\d{8}T\d{6})_")


def parse_date(stem):
    """Extract (date_str, hour_utc) from S2 scene stem."""
    m = S2_DATE_RE.search(stem)
    if not m:
        return None, None
    dt = m.group(1)
    date_str = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
    hour_utc = int(dt[9:11])
    return date_str, hour_utc


def fetch_era5(lat, lon, date_str, hour_utc, retries=3):
    """Fetch ERA5 hourly wind speed and temperature from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            hours = data["hourly"]["time"]
            temps = data["hourly"]["temperature_2m"]
            winds = data["hourly"]["wind_speed_10m"]
            target = f"{date_str}T{hour_utc:02d}:00"
            if target in hours:
                idx = hours.index(target)
            else:
                idx = min(range(len(hours)),
                          key=lambda i: abs(int(hours[i][11:13]) - hour_utc))
            return winds[idx], temps[idx]
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Fetch meteorological data for all chips")
    parser.add_argument("--split_log", default=SPLIT_LOG)
    parser.add_argument("--provenance_csv", default=PROVENANCE_CSV)
    parser.add_argument("--out_csv",
                        default=os.path.join(LLINKAS, "reference/met_data.csv"))
    parser.add_argument("--no_fetch", action="store_true",
                        help="Skip API fetch (just print summary of existing file)")
    args = parser.parse_args()

    if args.no_fetch and os.path.exists(args.out_csv):
        print(f"Loading existing met data from {args.out_csv}")
        _print_summary(args.out_csv)
        return

    # ── Build scene list ─────────────────────────────────────────────────
    # From split_log: Roboflow chips have S2 scene stems
    scene_to_chips = {}  # (scene_stem, region) → [chip_stem, ...]
    chip_meta = {}       # chip_stem → {sza_bin, split, ...}

    with open(args.split_log) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["stem"]
            chip_stem = row.get("chip_stem", stem)
            sza_bin = row["sza_bin"]
            split = row["split"]

            chip_meta[chip_stem] = {
                "sza_bin": sza_bin, "split": split, "source": "roboflow"
            }

            if not stem.startswith("fisser_"):
                # Infer region from tif_path
                tif_path = row.get("tif_path", "")
                region = "KQ" if "/KQ/" in tif_path else "SK" if "/SK/" in tif_path else "unknown"
                key = (stem, region)
                scene_to_chips.setdefault(key, []).append(chip_stem)

    # From provenance audit: Fisser chips
    if os.path.exists(args.provenance_csv):
        with open(args.provenance_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gidx = row["global_index"]
                chip_stem = f"fisser_{int(gidx):04d}"
                region = row["region"]
                tif_path = row["tif_path"]
                # Parse scene stem from tif filename
                basename = os.path.basename(tif_path)
                m = S2_DATE_RE.search(basename)
                if m:
                    scene_stem = basename.split("_pB")[0]  # strip chip suffix
                    key = (scene_stem, region)
                    scene_to_chips.setdefault(key, []).append(chip_stem)
                    chip_meta[chip_stem] = {
                        "sza_bin": "sza_lt65", "split": "", "source": "fisser"
                    }

    print(f"Unique scenes to query: {len(scene_to_chips)}")
    print(f"Total chips mapped:     {sum(len(v) for v in scene_to_chips.values())}")

    # ── Fetch met data per scene ─────────────────────────────────────────
    scene_met = {}
    n = len(scene_to_chips)
    fetched = 0

    for i, ((scene_stem, region), chips) in enumerate(scene_to_chips.items()):
        date_str, hour_utc = parse_date(scene_stem)
        if date_str is None:
            scene_met[(scene_stem, region)] = (None, None, date_str, hour_utc)
            continue

        lat, lon = REGION_COORDS.get(region, (68.0, -35.0))
        wind, temp = fetch_era5(lat, lon, date_str, hour_utc)
        scene_met[(scene_stem, region)] = (wind, temp, date_str, hour_utc)
        fetched += 1

        if (fetched) % 20 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}] fetched={fetched}  wind={wind}  temp={temp}")
        time.sleep(0.15)

    # ── Write per-chip CSV ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "chip_stem", "scene_stem", "source", "sza_bin", "split",
        "date", "hour_utc", "region",
        "wind_speed_10m", "temp_2m", "pass_wind", "pass_temp", "pass_both"
    ]

    rows_out = []
    for (scene_stem, region), chips in scene_to_chips.items():
        wind, temp, date_str, hour_utc = scene_met[(scene_stem, region)]
        pass_wind = (wind <= WIND_MAX) if wind is not None else None
        pass_temp = (temp > TEMP_MIN) if temp is not None else None
        pass_both = (pass_wind and pass_temp) if (pass_wind is not None and pass_temp is not None) else None

        for chip_stem in chips:
            meta = chip_meta.get(chip_stem, {})
            rows_out.append({
                "chip_stem": chip_stem,
                "scene_stem": scene_stem,
                "source": meta.get("source", ""),
                "sza_bin": meta.get("sza_bin", ""),
                "split": meta.get("split", ""),
                "date": date_str or "",
                "hour_utc": hour_utc if hour_utc is not None else "",
                "region": region,
                "wind_speed_10m": f"{wind:.2f}" if wind is not None else "",
                "temp_2m": f"{temp:.2f}" if temp is not None else "",
                "pass_wind": "" if pass_wind is None else str(pass_wind),
                "pass_temp": "" if pass_temp is None else str(pass_temp),
                "pass_both": "" if pass_both is None else str(pass_both),
            })

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nWrote {len(rows_out)} chip records to {args.out_csv}")
    _print_summary(args.out_csv)


def _print_summary(csv_path):
    """Print summary statistics from met_data.csv."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\n{'='*60}")
    print("METEOROLOGICAL DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total chip records: {len(rows)}")

    # Count by sza_bin
    by_bin = {}
    for r in rows:
        b = r["sza_bin"]
        by_bin.setdefault(b, {"total": 0, "has_met": 0, "pass_wind": 0, "pass_temp": 0, "pass_both": 0})
        by_bin[b]["total"] += 1
        if r["wind_speed_10m"]:
            by_bin[b]["has_met"] += 1
            if r["pass_wind"] == "True":
                by_bin[b]["pass_wind"] += 1
            if r["pass_temp"] == "True":
                by_bin[b]["pass_temp"] += 1
            if r["pass_both"] == "True":
                by_bin[b]["pass_both"] += 1

    print(f"\n{'SZA Bin':<15} {'Total':>6} {'HasMet':>7} {'Wind<=15':>9} {'Temp>0':>7} {'Both':>6}")
    print("-" * 55)
    for b in sorted(by_bin.keys()):
        d = by_bin[b]
        print(f"{b:<15} {d['total']:>6} {d['has_met']:>7} {d['pass_wind']:>9} {d['pass_temp']:>7} {d['pass_both']:>6}")

    # Wind/temp stats
    winds = [float(r["wind_speed_10m"]) for r in rows if r["wind_speed_10m"]]
    temps = [float(r["temp_2m"]) for r in rows if r["temp_2m"]]
    if winds:
        w = np.array(winds)
        print(f"\nWind speed: mean={w.mean():.1f}, median={np.median(w):.1f}, max={w.max():.1f} m/s")
        print(f"  Chips with wind > 15 m/s: {int((w > WIND_MAX).sum())} ({(w > WIND_MAX).sum()/len(w)*100:.1f}%)")
    if temps:
        t = np.array(temps)
        print(f"Temperature: mean={t.mean():.1f}, median={np.median(t):.1f}, max={t.max():.1f} C")
        print(f"  Chips with temp > 0 C:    {int((t > TEMP_MIN).sum())} ({(t > TEMP_MIN).sum()/len(t)*100:.1f}%)")

    # Fail-both count
    fail_wind = sum(1 for r in rows if r["pass_wind"] == "False")
    fail_temp = sum(1 for r in rows if r["pass_temp"] == "False")
    fail_either = sum(1 for r in rows if r["pass_wind"] == "False" or r["pass_temp"] == "False")
    print(f"\nChips failing wind threshold:    {fail_wind}")
    print(f"Chips failing temp threshold:    {fail_temp}")
    print(f"Chips failing either threshold:  {fail_either}")


if __name__ == "__main__":
    main()
