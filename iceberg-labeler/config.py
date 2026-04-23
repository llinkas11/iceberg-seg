"""
Central configuration for iceberg-labeler.

Edit the values in this file before running the server.
All paths can be absolute or relative to the project root.
"""

import os

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"   # 0.0.0.0 = accept connections from any machine on the network
PORT = 8000

# ── Admin ─────────────────────────────────────────────────────────────────────
# The admin token is used to access /admin endpoints and the admin dashboard.
# Change this before sharing the app with labelers.
ADMIN_TOKEN = "change-me-before-deploying"

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./iceberg_labels.db"

# ── Chip image storage ────────────────────────────────────────────────────────
# Absolute path to the directory that contains the chip PNG images.
# The import script copies/symlinks PNGs here, or you can point it directly
# to your chips/<region>/<sza_bin>/pngs/ folder.
# The server reads images from this path at runtime.
CHIPS_DIR = os.path.join(os.path.dirname(__file__), "data", "chips")

# ── Source data (used only by scripts/import_chips.py) ───────────────────────
# Root of the S2-iceberg-areas checkout — adjust if your layout differs.
IDS_ROOT = os.path.join(
    os.path.expanduser("~"),
    "Library", "CloudStorage", "OneDrive-BowdoinCollege",
    "Desktop", "IDS2026", "S2-iceberg-areas"
)

# Directories under IDS_ROOT that hold processed chips.
# Structure expected: CHIPS_SOURCE_ROOT/<region>/<sza_bin>/tifs/*.tif
#                                                          pngs/*.png
CHIPS_SOURCE_ROOT = os.path.join(IDS_ROOT, "chips")

# Directory that holds per-chip GeoPackage predictions.
# Structure expected: GPKG_SOURCE_ROOT/<REGION>/<sza_bin>/unet/gpkgs/<stem>_icebergs.gpkg
GPKG_SOURCE_ROOT = os.path.join(IDS_ROOT, "area_comparison")
