"""
Chip retrieval router.

Labelers use GET /api/chips/next to get their next unfinished chip.
The server picks the oldest pending assignment for the authenticated labeler.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse, Response

# Local cache directory to avoid OneDrive on-demand download timeouts
_IMAGE_CACHE_DIR = Path("/tmp/iceberg-labeler-cache/images")
_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db
from models import Labeler, Chip, Prediction, Assignment
from schemas import ChipWithPredictions, PredictionOut, ChipOut

router = APIRouter(prefix="/api/chips", tags=["chips"])


def _require_labeler(authorization: str = Header(...),
                     db: Session = Depends(get_db)) -> Labeler:
    """Dependency: validate Bearer token and return the Labeler."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token   = authorization.split(" ", 1)[1].strip()
    labeler = db.query(Labeler).filter(Labeler.token == token).first()
    if not labeler:
        raise HTTPException(401, "Invalid token")
    return labeler


def _chip_to_response(chip: Chip, assignment: Assignment) -> ChipWithPredictions:
    preds = [
        PredictionOut(
            id           = p.id,
            class_name   = p.class_name,
            pixel_coords = json.loads(p.pixel_coords),
            area_m2      = p.area_m2,
        )
        for p in chip.predictions
    ]
    return ChipWithPredictions(
        id               = chip.id,
        filename         = chip.filename,
        region           = chip.region,
        sza_bin          = chip.sza_bin,
        width            = chip.width,
        height           = chip.height,
        prediction_count = chip.prediction_count,
        predictions      = preds,
        assignment_id    = assignment.id,
        assignment_status= assignment.status,
    )


@router.get("/next", response_model=Optional[ChipWithPredictions])
def get_next_chip(labeler: Labeler = Depends(_require_labeler),
                  db: Session = Depends(get_db)):
    """
    Return the next pending assignment for this labeler.
    Returns null (HTTP 204) if no chips remain in their queue.
    """
    assignment = (
        db.query(Assignment)
        .filter(
            Assignment.labeler_id == labeler.id,
            Assignment.status.in_(["pending", "in_progress"])
        )
        .order_by(Assignment.assigned_at)
        .first()
    )
    if not assignment:
        return None

    # Mark as in_progress so admin can track who is actively working
    if assignment.status == "pending":
        assignment.status = "in_progress"
        db.commit()

    chip = db.query(Chip).filter(Chip.id == assignment.chip_id).first()
    return _chip_to_response(chip, assignment)


@router.get("/assigned", response_model=list[ChipOut])
def get_assigned_chips(labeler: Labeler = Depends(_require_labeler),
                       db: Session = Depends(get_db)):
    """Return summary of all chips assigned to this labeler."""
    assignments = (
        db.query(Assignment)
        .filter(Assignment.labeler_id == labeler.id)
        .order_by(Assignment.assigned_at)
        .all()
    )
    results = []
    for asgn in assignments:
        chip = db.query(Chip).filter(Chip.id == asgn.chip_id).first()
        results.append(ChipOut(
            id               = chip.id,
            filename         = chip.filename,
            region           = chip.region,
            sza_bin          = chip.sza_bin,
            width            = chip.width,
            height           = chip.height,
            prediction_count = chip.prediction_count,
        ))
    return results


@router.get("/{chip_id}", response_model=ChipWithPredictions)
def get_chip(chip_id: int,
             labeler: Labeler = Depends(_require_labeler),
             db: Session = Depends(get_db)):
    """Fetch a specific chip (must be assigned to this labeler or labeler is admin)."""
    chip = db.query(Chip).filter(Chip.id == chip_id).first()
    if not chip:
        raise HTTPException(404, "Chip not found")

    assignment = (
        db.query(Assignment)
        .filter(Assignment.chip_id == chip_id,
                Assignment.labeler_id == labeler.id)
        .first()
    )
    if not assignment and not labeler.is_admin:
        raise HTTPException(403, "Chip is not assigned to you")

    if not assignment:
        # Admin viewing unassigned chip — create a dummy wrapper
        class _FakeAssignment:
            id     = -1
            status = "admin-preview"
        assignment = _FakeAssignment()

    return _chip_to_response(chip, assignment)


# ── Image file endpoint ───────────────────────────────────────────────────────

@router.get("/image/{chip_id}")
def get_chip_image(chip_id: int, db: Session = Depends(get_db)):
    """Serve the PNG image for a chip (no auth — images are not sensitive)."""
    chip = db.query(Chip).filter(Chip.id == chip_id).first()
    if not chip:
        raise HTTPException(404, "Chip not found")
    if not os.path.exists(chip.image_path):
        raise HTTPException(404, f"Image file missing: {chip.image_path}")

    # Serve from local cache to avoid OneDrive on-demand download timeouts
    cache_key = hashlib.md5(chip.image_path.encode()).hexdigest()
    cached = _IMAGE_CACHE_DIR / f"{cache_key}.png"
    if not cached.exists():
        shutil.copy2(chip.image_path, cached)

    return FileResponse(str(cached), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
