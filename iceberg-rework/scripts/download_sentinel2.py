"""
Sentinel-2 L1C Search & Download Script
========================================
For: Iceberg detection at high solar zenith angles
Study regions: Kangerlussuaq (KQ) and Sermilik (SK) Fjords, Greenland
Data source: Copernicus Data Space Ecosystem (free, ESA)
Paper reference: Fisser et al. 2024, Rezvanbehbahani et al. 2020

SETUP (run once in terminal):
    pip install requests pandas pysolar

USAGE:
    1. Fill in your COPERNICUS_USER and COPERNICUS_PASSWORD below
    2. Run: python download_sentinel2.py
    3. Images will be saved to ./sentinel2_downloads/ organised by SZA bin

WHAT THIS DOES:
    - Searches for Sentinel-2 L1C scenes over KQ and SK fjords
    - Filters Sept-Nov (high solar zenith angle season)
    - Filters cloud cover < 20%
    - Prints solar zenith angle for each scene from metadata
    - Downloads the full .SAFE zip (contains all bands incl. B8 NIR)
    - Organises downloads into SZA bins matching the paper evaluation bins:
        sentinel2_downloads/
          KQ/
            sza_lt65/      < 65°  (summer baseline)
            sza_65_70/     65–70° (marginal)
            sza_70_75/     70–75° (high)
            sza_gt75/      > 75°  (very high — Oct/Nov primary target)
          SK/
            ...
          scene_catalogue.csv
"""

import os
import math
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_token_lock = threading.Lock()
_token_cache = {"token": None}

# ─────────────────────────────────────────────
# 1. YOUR CREDENTIALS — fill these in
# ─────────────────────────────────────────────
COPERNICUS_USER = "smishra@bowdoin.edu"       # your dataspace.copernicus.eu email
COPERNICUS_PASSWORD = "7nn~+/X-7?auPSQ"        # your password

# ─────────────────────────────────────────────
# 2. STUDY AREAS (from Fisser's GeoPackage, converted to WGS84 bounding boxes)
# ─────────────────────────────────────────────
# KQ = Kangerlussuaq Fjord (~68°N, ~-32°E) — primary target per Fisser & Dr. Fisser email
# SK = Sermilik Fjord (~66°N, ~-38°E) — secondary target
AREAS_OF_INTEREST = {
    "KQ": "POLYGON((-33.5 67.5, -30.5 67.5, -30.5 69.0, -33.5 69.0, -33.5 67.5))",
    "SK": "POLYGON((-39.0 65.0, -36.0 65.0, -36.0 67.0, -39.0 67.0, -39.0 65.0))",
}

# ─────────────────────────────────────────────
# 3. SEARCH PARAMETERS
# ─────────────────────────────────────────────
# Search month-by-month so the per-query cap doesn't fill up with Sept scenes.
# SZA at KQ/SK latitudes (~66-68°N):
#   September  → ~65-70°  (sza_65_70  bin)
#   October    → ~70-76°  (sza_70_75 / sza_gt75 bins) ← primary target
#   November   → ~76-83°  (sza_gt75 bin)             ← primary target
SEARCH_WINDOWS = [
    # ── Early September (SZA ~58–62° at KQ/SK → sza_lt65) ──────────────────
    ("2015-09-01", "2015-09-14"),
    ("2016-09-01", "2016-09-14"),
    ("2017-09-01", "2017-09-14"),
    ("2018-09-01", "2018-09-14"),
    ("2019-09-01", "2019-09-14"),
    ("2020-09-01", "2020-09-14"),
    ("2021-09-01", "2021-09-14"),
    ("2022-09-01", "2022-09-14"),
    ("2023-09-01", "2023-09-14"),
    ("2024-09-01", "2024-09-14"),
    # ── Late September (SZA ~65–70° at KQ → sza_65_70) ──────────────────────
    ("2015-09-15", "2015-09-30"),
    ("2016-09-15", "2016-09-30"),
    ("2017-09-15", "2017-09-30"),
    ("2018-09-15", "2018-09-30"),
    ("2019-09-15", "2019-09-30"),
    ("2020-09-15", "2020-09-30"),
    ("2021-09-15", "2021-09-30"),
    ("2022-09-15", "2022-09-30"),
    ("2023-09-15", "2023-09-30"),
    ("2024-09-15", "2024-09-30"),
    # ── Early October (SZA ~70–75° → sza_70_75) ─────────────────────────────
    ("2015-10-01", "2015-10-31"),
    ("2016-10-01", "2016-10-31"),
    ("2017-10-01", "2017-10-31"),
    ("2018-10-01", "2018-10-31"),
    ("2019-10-01", "2019-10-31"),
    ("2020-10-01", "2020-10-31"),
    ("2021-10-01", "2021-10-31"),
    ("2022-10-01", "2022-10-31"),
    ("2023-10-01", "2023-10-31"),
    ("2024-10-01", "2024-10-31"),
    # ── November (SZA >75° → sza_gt75) ──────────────────────────────────────
    ("2015-11-01", "2015-11-30"),
    ("2016-11-01", "2016-11-30"),
    ("2017-11-01", "2017-11-30"),
    ("2018-11-01", "2018-11-30"),
    ("2019-11-01", "2019-11-30"),
    ("2020-11-01", "2020-11-30"),
    ("2021-11-01", "2021-11-30"),
    ("2022-11-01", "2022-11-30"),
    ("2023-11-01", "2023-11-30"),
    ("2024-11-01", "2024-11-30"),
]
MAX_CLOUD_COVER = 20   # percent
MAX_RESULTS = 5        # per window per region

# ─────────────────────────────────────────────
# 4. OUTPUT DIRECTORY
# ─────────────────────────────────────────────
OUTPUT_DIR = "/mnt/research/v.gomezgilyaspik/students/smishra/sentinel2_downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════

def get_access_token(user, password):
    """Get OAuth2 token from Copernicus Data Space."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": user,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]


def search_sentinel2(aoi_wkt, start_date, end_date, max_cloud=20, max_results=50):
    """
    Search Copernicus catalogue for Sentinel-2 L1C scenes.
    Returns a list of product metadata dicts.
    """
    # L1C = Top-of-Atmosphere reflectance (what Fisser uses — no atmospheric correction)
    product_filter = "S2MSI1C"

    url = (
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq 'SENTINEL-2'"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')"
        f" and ContentDate/Start gt {start_date}T00:00:00.000Z"
        f" and ContentDate/Start lt {end_date}T23:59:59.000Z"
        f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'"
        f"   and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f"   and att/OData.CSC.StringAttribute/Value eq '{product_filter}')"
        f"&$orderby=ContentDate/Start asc"
        f"&$top={max_results}"
        f"&$expand=Attributes"
    )

    r = requests.get(url)
    r.raise_for_status()
    return r.json().get("value", [])


def extract_solar_zenith(product):
    """
    Extract mean solar zenith angle from product attributes.
    Falls back to pysolar calculation if not in metadata.
    """
    # Try to get from metadata attributes
    attrs = product.get("Attributes", [])
    for attr in attrs:
        if "zenith" in attr.get("Name", "").lower() or "sunZenith" in attr.get("Name", ""):
            return round(attr.get("Value", None), 2)

    # Fallback: calculate from acquisition time and scene center using pysolar
    try:
        from pysolar.solar import get_altitude
        name = product.get("Name", "")
        # Parse date from product name: S2B_MSIL1C_20211005T...
        date_str = name.split("_")[2][:8]  # "20211005"
        acq_date = datetime.strptime(date_str, "%Y%m%d")
        acq_date = acq_date.replace(hour=12, tzinfo=__import__("datetime").timezone.utc)

        # Use scene footprint center
        fp = product.get("GeoFootprint", {}).get("coordinates", [[[0, 0]]])[0]
        lats = [c[1] for c in fp]
        lons = [c[0] for c in fp]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        altitude_deg = get_altitude(center_lat, center_lon, acq_date)
        return round(90 - altitude_deg, 2)  # solar zenith = 90 - elevation
    except Exception:
        return None


def solar_zenith_from_name(name):
    """Quick estimate of solar zenith from acquisition date and KQ/SK latitude."""
    try:
        date_str = name.split("_")[2][:8]
        month = int(date_str[4:6])
        # Approximate SZA at Kangerlussuaq (68°N) by month
        # Based on Fisser 2024 Fig 1 data
        monthly_sza = {9: 68, 10: 76, 11: 83, 8: 60, 7: 48, 6: 44}
        return monthly_sza.get(month, "?")
    except Exception:
        return "?"


def get_sza_bin(sza):
    """
    Return folder name for a solar zenith angle, matching the paper's evaluation bins.
    Handles numeric SZA or monthly-estimate strings/ints.
    """
    try:
        sza = float(sza)
    except (TypeError, ValueError):
        return "sza_unknown"
    if sza < 65:
        return "sza_lt65"
    elif sza < 70:
        return "sza_65_70"
    elif sza < 75:
        return "sza_70_75"
    else:
        return "sza_gt75"


def download_product(product_id, product_name, token, output_dir):
    """Download a Sentinel-2 product zip file."""
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    out_path = os.path.join(output_dir, f"{product_name}.zip")
    if os.path.exists(out_path):
        print(f"  ✓ Already downloaded: {product_name}.zip")
        return out_path

    print(f"  ↓ Downloading {product_name} ...")
    session = requests.Session()
    session.headers.update(headers)
    r = session.get(url, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {pct:.1f}%", end="\r")
    print(f"  ✓ Saved: {out_path}")
    return out_path


# ═════════════════════════════════════════════
# MAIN SCRIPT
# ═════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Sentinel-2 Search & Download — Greenland Iceberg Study")
    print("=" * 60)

    # ── Step 1: Authenticate ──
    print("\n[1/3] Authenticating with Copernicus Data Space...")
    try:
        token = get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD)
        print("  ✓ Token obtained")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print("  → Check your email/password at dataspace.copernicus.eu")
        return

    # ── Step 2: Search ──
    print("\n[2/3] Searching catalogue...")
    all_results = []

    for region_name, aoi_wkt in AREAS_OF_INTEREST.items():
        for start, end in SEARCH_WINDOWS:
            print(f"  Searching {region_name} | {start[:7]} to {end[:7]}...")
            try:
                products = search_sentinel2(aoi_wkt, start, end,
                                            max_cloud=MAX_CLOUD_COVER,
                                            max_results=MAX_RESULTS)
                print(f"    → {len(products)} scenes found")
                for p in products:
                    sza_est = solar_zenith_from_name(p.get("Name", ""))
                    all_results.append({
                        "id": p["Id"],
                        "name": p["Name"],
                        "region": region_name,
                        "date": p.get("ContentDate", {}).get("Start", "")[:10],
                        "cloud_cover": next(
                            (a["Value"] for a in p.get("Attributes", [])
                             if a.get("Name") == "cloudCover"), "?"
                        ),
                        "sza_estimate": sza_est,
                        "size_mb": round(p.get("ContentLength", 0) / 1e6, 1),
                    })
            except Exception as e:
                print(f"    ✗ Search error: {e}")

    if not all_results:
        print("\n  No results found. Try broadening your search parameters.")
        return

    # ── Step 3: Display results ──
    df = pd.DataFrame(all_results).drop_duplicates(subset="id")
    df = df.sort_values(["region", "date"])

    print(f"\n{'─'*60}")
    print(f"FOUND {len(df)} UNIQUE SCENES")
    print(f"{'─'*60}")
    print(df[["region", "date", "cloud_cover", "sza_estimate", "size_mb", "name"]].to_string(index=False))

    # Save catalogue to CSV for reference
    csv_path = os.path.join(OUTPUT_DIR, "scene_catalogue.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Scene catalogue saved to: {csv_path}")

    # ── Step 4: Download prompt ──
    print(f"\n[3/3] Download")
    print("  Full Sentinel-2 .SAFE files are ~800MB each (all bands).")
    print("  Options:")
    print("    [A] Download ALL scenes listed above")
    print("    [B] Download scenes from ONE region (KQ or SK)")
    print("    [C] Download a SINGLE scene by entering its name")
    print("    [N] Skip download for now (catalogue already saved)")

    choice = input("\n  Your choice [A/B/C/N]: ").strip().upper()

    if choice == "N":
        print("  → Skipping download. Run script again to download later.")
        print(f"  → Scene list saved to {csv_path}")
        return

    scenes_to_download = []

    if choice == "A":
        scenes_to_download = df.to_dict("records")

    elif choice == "B":
        region = input("  Enter region (KQ or SK): ").strip().upper()
        scenes_to_download = df[df["region"] == region].to_dict("records")

    elif choice == "C":
        print("  Enter scene name (copy from table above):")
        name_input = input("  → ").strip()
        match = df[df["name"].str.contains(name_input[:20])]
        if match.empty:
            print("  ✗ Scene not found in results.")
            return
        scenes_to_download = match.to_dict("records")

    # Cap at MAX_PER_BIN per (region, sza_bin) to avoid over-downloading
    MAX_PER_BIN = 30
    scenes_df = pd.DataFrame(scenes_to_download)
    scenes_df["sza_bin"] = scenes_df["sza_estimate"].apply(get_sza_bin)
    scenes_df = (
        scenes_df
        .groupby(["region", "sza_bin"], group_keys=False)
        .apply(lambda g: g.head(MAX_PER_BIN))
        .reset_index(drop=True)
    )
    scenes_to_download = scenes_df.to_dict("records")

    # Show SZA bin summary before downloading
    scenes_df = pd.DataFrame(scenes_to_download)
    scenes_df["sza_bin"] = scenes_df["sza_estimate"].apply(get_sza_bin)
    bin_counts = scenes_df.groupby(["region", "sza_bin"]).size().reset_index(name="count")
    print(f"\n  SZA bin breakdown:")
    print(bin_counts.to_string(index=False))
    total_gb = scenes_df["size_mb"].sum() / 1024
    print(f"\n  Total download size: ~{total_gb:.1f} GB")
    confirm = input(f"\n  Download {len(scenes_to_download)} scene(s)? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  → Cancelled. Catalogue already saved.")
        return

    N_WORKERS = 4  # parallel downloads — Copernicus rate-limits beyond ~4-6
    TOKEN_REFRESH_INTERVAL = 480  # seconds (tokens last ~600s, refresh every 8 min)

    import time
    shared_token = {"value": get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD),
                    "ts": time.time()}

    def get_shared_token():
        with _token_lock:
            if time.time() - shared_token["ts"] > TOKEN_REFRESH_INTERVAL:
                shared_token["value"] = get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD)
                shared_token["ts"] = time.time()
            return shared_token["value"]

    def download_one(scene):
        sza_bin = get_sza_bin(scene["sza_estimate"])
        out_dir = os.path.join(OUTPUT_DIR, scene["region"], sza_bin)
        os.makedirs(out_dir, exist_ok=True)
        try:
            token = get_shared_token()
            download_product(scene["id"], scene["name"], token, out_dir)
            return scene["name"], None
        except Exception as e:
            return scene["name"], str(e)

    print(f"\n  Downloading {len(scenes_to_download)} scene(s) with {N_WORKERS} parallel workers...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(download_one, s): s for s in scenes_to_download}
        for fut in as_completed(futures):
            name, err = fut.result()
            if err:
                print(f"  ✗ Failed: {name[:60]} — {err}")

    print("\n✓ Done!")
    print(f"  Downloads in: {OUTPUT_DIR}/{{region}}/{{sza_bin}}/")
    print("\n  Next steps:")
    print("  1. Run chip_sentinel2.py to extract + tile the .SAFE files into 256×256 chips")
    print("  2. Run predict_tifs.py on the chips to get UNet++ iceberg delineations")
    print("  3. Compare UNet++ vs Fisser 0.12 NIR threshold per SZA bin")


if __name__ == "__main__":
    main()