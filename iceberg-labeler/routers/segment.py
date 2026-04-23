"""
SAM segmentation router.

Provides interactive click-to-segment: labeler clicks a point on the chip
image, and SAM returns a polygon outlining the object at that location.

Endpoints:
    POST /api/segment/predict  — click → polygon
    GET  /api/segment/status   — is SAM loaded?
"""

import os
import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from database import get_db
from models import Labeler, Chip
from schemas import SegmentRequest, SegmentResponse
from sam_predictor import get_sam_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/segment", tags=["segmentation"])


def _require_labeler(authorization: str = Header(...),
                     db: Session = Depends(get_db)) -> Labeler:
    """Dependency: validate Bearer token and return the Labeler."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    labeler = db.query(Labeler).filter(Labeler.token == token).first()
    if not labeler:
        raise HTTPException(401, "Invalid token")
    return labeler


@router.get("/status")
def segment_status():
    """Check whether SAM is loaded and return model info."""
    sam = get_sam_service()
    if sam is None:
        return {
            "available": False,
            "error": "SAM service not initialized",
        }
    return sam.model_info


@router.post("/predict", response_model=SegmentResponse)
def segment_predict(
    req: SegmentRequest,
    labeler: Labeler = Depends(_require_labeler),
    db: Session = Depends(get_db),
):
    """
    Click-to-segment: given point coordinates on a chip image,
    return a polygon outlining the object at that location.
    """
    sam = get_sam_service()
    if sam is None or not sam.is_available:
        raise HTTPException(503, "SAM model not available")

    # Look up chip
    chip = db.query(Chip).filter(Chip.id == req.chip_id).first()
    if not chip:
        raise HTTPException(404, "Chip not found")

    # Resolve image path (check cached version first)
    image_path = chip.image_path
    cache_path = f"/tmp/iceberg-labeler-cache/images/{chip.id}.png"
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        image_path = cache_path
    elif not os.path.exists(image_path):
        raise HTTPException(404, f"Chip image not found: {image_path}")

    # Validate input
    if len(req.point_coords) == 0:
        raise HTTPException(400, "At least one point is required")
    if len(req.point_coords) != len(req.point_labels):
        raise HTTPException(400, "point_coords and point_labels must have same length")
    for label in req.point_labels:
        if label not in (0, 1):
            raise HTTPException(400, "point_labels must be 0 (background) or 1 (foreground)")

    try:
        result = sam.predict(
            chip_id=req.chip_id,
            image_path=image_path,
            point_coords=req.point_coords,
            point_labels=req.point_labels,
            multimask=req.multimask,
        )
    except Exception as e:
        logger.error("SAM prediction failed for chip %d: %s", req.chip_id, e)
        raise HTTPException(500, f"SAM prediction failed: {e}")

    return SegmentResponse(
        polygon=result["polygon"],
        confidence=result["confidence"],
        area_px=result["area_px"],
    )
