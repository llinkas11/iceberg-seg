# Iceberg Labeler

A web application for validating UNet++ iceberg predictions from Sentinel-2 satellite imagery.
Labelers see each 256×256 chip with the model's predicted polygons overlaid, then accept,
reject, or edit them. All results are stored in a local SQLite database and exportable
as COCO JSON (for model retraining) or CSV (for analysis).

---

## Quick Start (for the researcher running the server)

### 1. Install dependencies

```bash
cd iceberg-labeler
pip install -r requirements.txt
```

> **Note:** `rasterio` and `geopandas` must be installed in the same environment
> you use for the main IDS2026 research code. If using the `iceberg-unet` conda
> environment, activate it first.

### 2. Set the admin token

Edit `config.py` and change `ADMIN_TOKEN` to a secret string you choose:

```python
ADMIN_TOKEN = "my-secret-token-2026"
```

Keep this private — it controls the admin dashboard and data exports.

### 3. Import chips and predictions

```bash
# Import from the default IDS2026 paths (set in config.py):
python scripts/import_chips.py

# Or specify paths explicitly:
python scripts/import_chips.py \
    --chips-root /path/to/S2-iceberg-areas/chips \
    --gpkg-root  /path/to/S2-iceberg-areas/area_comparison

# Filter to specific region/SZA bin during testing:
python scripts/import_chips.py --region kq --sza-bin sza_65_70 --dry-run
# (remove --dry-run to actually import)
```

### 4. Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The `--host 0.0.0.0` flag makes the server accessible to other machines on the
same network. Your labelers connect to `http://<your-ip>:8000`.

Find your IP address:
- macOS: `System Settings → Network → Wi-Fi → Details`
- Terminal: `ipconfig getifaddr en0`

### 5. Open the admin dashboard

Visit `http://localhost:8000/admin` in your browser.
Enter the `ADMIN_TOKEN` from `config.py`.

### 6. Assign chips to labelers

Wait for labelers to register (they open the labeler URL and enter their name).
Then use the admin dashboard → **Assign Chips** to distribute work.

Or from the terminal:
```bash
python scripts/assign_chips.py assign --labeler "Alex Smith" --n 100
python scripts/assign_chips.py distribute    # spread all chips evenly among all labelers
python scripts/assign_chips.py summary
```

---

## For Labelers (students / image testers)

1. Open `http://<server-ip>:8000` in your browser (Chrome or Firefox recommended).
2. Click **New labeler**, enter your name, click **Register & Start**.
3. **Save your token** — copy it somewhere safe. You'll need it to log back in.
4. Start labeling!

### Labeling workflow

Each chip shows the UNet++ predicted polygons overlaid on the satellite image:
- **Blue polygons** = predicted icebergs
- **Orange polygons** = predicted shadows

For each chip, choose one of:

| Button | Meaning |
|--------|---------|
| **✓ Accept All** | All predictions are correct — accept every polygon as-is |
| **✗ No Icebergs** | The chip contains no icebergs — reject all predictions |
| **↑ Submit Edit** | You reviewed each polygon individually (see below) |
| **→ Skip** | Chip is too ambiguous — skip for now |

**Individual polygon review** (when predictions are partially right):
- Click any polygon on the image or in the sidebar list to select it
- Click **✓** (accept) or **✗** (reject) per polygon
- Use **Draw Iceberg / Draw Shadow** to add polygons the model missed
- When every polygon has a decision, click **↑ Submit Edit**

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `A` | Accept all |
| `X` | Reject all (no icebergs) |
| `S` | Skip chip |
| `I` | Draw new iceberg polygon |
| `H` | Draw new shadow polygon |
| `Esc` | Cancel drawing |
| `↑ / ↓` | Navigate polygon list |
| `Enter` | Accept selected polygon |
| `Delete` | Reject selected polygon |

---

## Admin Reference

### Dashboard (`/admin`)

| Section | Purpose |
|---------|---------|
| Stats cards | Total chips, complete, pending at a glance |
| Assign Chips | Distribute chips to labelers; filter by region/SZA bin |
| Labeler Progress | Per-labeler completion table |
| Export | Download COCO JSON or CSV |
| Chip Browser | Inspect individual chips and their assignment status |

### Exports

**COCO JSON** (`/api/export/coco`):
- Standard COCO instance segmentation format
- One annotation per accepted/added/modified polygon
- Includes `region` and `sza_bin` metadata per image
- Ready for direct use with Roboflow, CVAT, or custom training pipelines

**CSV** (`/api/export/csv`):
- One row per completed chip
- Columns: chip filename, region, SZA bin, labeler, verdict, n_accepted,
  n_rejected, n_added, n_modified, prediction_count, notes, submitted_at

### Database backup

The entire database is a single file: `iceberg_labels.db` in the project root.
Back it up by copying it to OneDrive:

```bash
cp iceberg_labels.db ~/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/IDS2026/iceberg_labels_backup_$(date +%Y%m%d).db
```

---

## Architecture

```
iceberg-labeler/
├── app.py                  # FastAPI entry point
├── config.py               # ← edit before running (admin token, paths)
├── database.py             # SQLAlchemy + SQLite (WAL mode)
├── models.py               # ORM: Chip, Prediction, Labeler, Assignment, …
├── schemas.py              # Pydantic request/response schemas
├── routers/
│   ├── auth.py             # POST /api/auth/register, /login
│   ├── chips.py            # GET /api/chips/next, /{id}, /image/{id}
│   ├── annotations.py      # POST /api/annotations
│   ├── admin.py            # GET/POST /api/admin/*  (admin token required)
│   └── export.py           # GET /api/export/coco, /csv  (admin token required)
├── static/
│   ├── index.html          # Labeler UI (login + annotation canvas)
│   ├── admin.html          # Admin dashboard
│   ├── js/
│   │   ├── annotator.js    # Leaflet + Leaflet.draw annotation logic
│   │   └── admin.js        # Admin dashboard logic
│   └── css/style.css
├── scripts/
│   ├── import_chips.py     # Import PNGs + GeoPackage predictions → SQLite
│   └── assign_chips.py     # CLI for chip assignment
└── data/
    └── chips/              # PNG files served to labelers (populated by import)
```

### Data model

```
Chip ──< Prediction          (model's predicted polygons per chip)
Chip ──< Assignment ──── Labeler
Assignment ──< AnnotationResult ──< PolygonDecision
```

- A chip can be assigned to multiple labelers (for inter-rater agreement)
- Each PolygonDecision records the final outcome per polygon:
  `accepted | rejected | added | modified`

---

## Troubleshooting

**"Image file missing" error**
The PNG was imported but has since moved. Re-run `import_chips.py` or manually
copy the PNG to `data/chips/`.

**Labeler can't connect**
- Confirm the server is running with `--host 0.0.0.0`
- Confirm labeler is on the same network
- Check firewall: `sudo lsof -i :8000`
- For remote access: use `ngrok http 8000` to create a public tunnel

**Polygon coordinates look wrong**
Run `python scripts/import_chips.py --dry-run --region kq` and check that
the GeoPackage path is found. The most common cause is the `GPKG_SOURCE_ROOT`
pointing to the wrong directory — adjust in `config.py`.

**Database locked**
SQLite WAL mode is enabled, so concurrent reads/writes should be safe.
If you see a lock error, check that no other process has the DB open exclusively.

---

## Extending the tool

- **Add metadata display** (SZA value, date): add fields to the `chips` table
  during import and display them in `index.html`.
- **Add locator map**: include a per-chip locator map PNG in `data/chips/` and
  display it as a second Leaflet layer toggle.
- **Inter-rater agreement**: assign the same chip to multiple labelers, then
  compare `PolygonDecision` records across `assignment_id`s.
- **Filtered exports**: use query parameters on `/api/export/coco?region=kq&sza_bin=sza_65_70`
  to export a specific subset.
