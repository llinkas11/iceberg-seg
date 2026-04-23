"""
Annotation submission router.

POST /api/annotations submits a labeler's verdict for one chip.
The server validates the submission, stores it, and returns whether a next
chip is available.
"""

import json
import math
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from database import get_db
from models import (Labeler, Chip, Prediction, Assignment,
                    AnnotationResult, PolygonDecision)
from schemas import AnnotationSubmit, AnnotationResponse

router = APIRouter(prefix="/api/annotations", tags=["annotations"])

VALID_VERDICTS  = {"accepted", "rejected", "edited", "skipped"}
VALID_ACTIONS   = {"accepted", "rejected", "added", "modified"}
VALID_CLASSES   = {"iceberg"}
VALID_TAGS      = {"sea-ice", "ambiguous", "cloud", "land-edge", "melange", "dark-water"}


def _require_labeler(authorization: str = Header(...),
                     db: Session = Depends(get_db)) -> Labeler:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token   = authorization.split(" ", 1)[1].strip()
    labeler = db.query(Labeler).filter(Labeler.token == token).first()
    if not labeler:
        raise HTTPException(401, "Invalid token")
    return labeler


def _polygon_area_px(coords: list[list[float]]) -> float:
    """Shoelace formula: pixel area of a polygon given [[x,y], ...] coords."""
    n = len(coords)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


@router.post("", response_model=AnnotationResponse)
def submit_annotation(
    body: AnnotationSubmit,
    labeler: Labeler = Depends(_require_labeler),
    db: Session = Depends(get_db),
):
    # ── Validate assignment ownership ─────────────────────────────────────────
    assignment = db.query(Assignment).filter(
        Assignment.id         == body.assignment_id,
        Assignment.chip_id    == body.chip_id,
        Assignment.labeler_id == labeler.id,
    ).first()
    if not assignment:
        raise HTTPException(403, "Assignment not found or does not belong to you")

    if assignment.status == "complete":
        raise HTTPException(409, "This chip has already been submitted")

    # ── Validate verdict ──────────────────────────────────────────────────────
    if body.verdict not in VALID_VERDICTS:
        raise HTTPException(400, f"verdict must be one of {VALID_VERDICTS}")

    # ── If 'accepted' or 'rejected', auto-generate polygon decisions ──────────
    chip        = db.query(Chip).filter(Chip.id == body.chip_id).first()
    predictions = db.query(Prediction).filter(
        Prediction.chip_id == body.chip_id
    ).all()
    pred_map    = {p.id: p for p in predictions}

    if body.verdict == "accepted":
        polygon_decisions = [
            {
                "prediction_id": p.id,
                "action":        "accepted",
                "class_name":    p.class_name,
                "pixel_coords":  json.loads(p.pixel_coords),
            }
            for p in predictions
        ]
    elif body.verdict == "rejected":
        polygon_decisions = [
            {
                "prediction_id": p.id,
                "action":        "rejected",
                "class_name":    p.class_name,
                "pixel_coords":  json.loads(p.pixel_coords),
            }
            for p in predictions
        ]
    elif body.verdict == "skipped":
        polygon_decisions = []
    else:
        # "edited" — use what the labeler sent
        polygon_decisions = []
        for dec in body.polygon_decisions:
            if dec.action not in VALID_ACTIONS:
                raise HTTPException(400, f"action must be one of {VALID_ACTIONS}")
            if dec.class_name not in VALID_CLASSES:
                raise HTTPException(400, f"class_name must be one of {VALID_CLASSES}")
            if dec.prediction_id and dec.prediction_id not in pred_map:
                raise HTTPException(400, f"prediction_id {dec.prediction_id} "
                                         "does not belong to this chip")
            polygon_decisions.append({
                "prediction_id": dec.prediction_id,
                "action":        dec.action,
                "class_name":    dec.class_name,
                "pixel_coords":  dec.pixel_coords,
            })

    # ── Persist ───────────────────────────────────────────────────────────────
    # Validate tags
    tags_str = ""
    if body.tags:
        invalid = set(body.tags) - VALID_TAGS
        if invalid:
            raise HTTPException(400, f"Invalid tags: {invalid}. Valid: {VALID_TAGS}")
        tags_str = ",".join(sorted(body.tags))

    result = AnnotationResult(
        assignment_id = assignment.id,
        chip_verdict  = body.verdict,
        notes         = body.notes or "",
        tags          = tags_str,
    )
    db.add(result)
    db.flush()  # get result.id before adding children

    for dec in polygon_decisions:
        coords   = dec["pixel_coords"]
        area_px  = _polygon_area_px(coords)
        pd_row   = PolygonDecision(
            result_id     = result.id,
            prediction_id = dec["prediction_id"],
            action        = dec["action"],
            class_name    = dec["class_name"],
            pixel_coords  = json.dumps(coords),
            area_px       = round(area_px, 2),
        )
        db.add(pd_row)

    # ── Update assignment status ──────────────────────────────────────────────
    assignment.status       = "complete" if body.verdict != "skipped" else "skipped"
    assignment.completed_at = datetime.utcnow()
    db.commit()

    # ── Check whether more chips remain in this labeler's queue ──────────────
    next_pending = db.query(Assignment).filter(
        Assignment.labeler_id == labeler.id,
        Assignment.status.in_(["pending", "in_progress"])
    ).first()

    return AnnotationResponse(
        annotation_result_id = result.id,
        next_chip_available  = next_pending is not None,
    )


@router.get("/my-progress")
def my_progress(labeler: Labeler = Depends(_require_labeler),
                db: Session = Depends(get_db)):
    """Return a summary of this labeler's annotation progress."""
    total    = db.query(Assignment).filter(
        Assignment.labeler_id == labeler.id
    ).count()
    complete = db.query(Assignment).filter(
        Assignment.labeler_id == labeler.id,
        Assignment.status     == "complete"
    ).count()
    skipped  = db.query(Assignment).filter(
        Assignment.labeler_id == labeler.id,
        Assignment.status     == "skipped"
    ).count()
    pending  = total - complete - skipped
    return {
        "total":    total,
        "complete": complete,
        "skipped":  skipped,
        "pending":  pending,
        "percent":  round(complete / total * 100, 1) if total else 0,
    }
