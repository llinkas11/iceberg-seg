"""
iceberg-labeler — FastAPI entry point.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import init_db
from routers import auth, chips, annotations, admin, export

app = FastAPI(
    title       = "Iceberg Labeler",
    description = "Web tool for validating UNet++ iceberg predictions",
    version     = "1.0.0",
)

# ── Create DB tables on startup ───────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(chips.router)
app.include_router(annotations.router)
app.include_router(admin.router)
app.include_router(export.router)

# ── Static files (JS, CSS) ────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── SPA catch-all routes ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(os.path.join(static_dir, "admin.html"))
