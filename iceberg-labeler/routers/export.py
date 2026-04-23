"""
Export router — produces COCO JSON and CSV exports.

All exports require admin token.
"""

import json
import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models import (Labeler, Chip, Prediction, Assignment,
                    AnnotationResult, PolygonDecision)
import config

router = APIRouter(prefix="/api/export", tags=["export"])


def _require_admin(authorization: str = Header(...),
                   db: Session = Depends(get_db)) -> Labeler:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token   = authorization.split(" ", 1)[1].strip()
    if token == config.ADMIN_TOKEN:
        return db.query(Labeler).filter(Labeler.is_admin == True).first() or \
               Labeler(name="admin", is_admin=True)
    labeler = db.query(Labeler).filter(Labeler.token == token,
                                       Labeler.is_admin == True).first()
    if not labeler:
        raise HTTPException(403, "Admin access required")
    return labeler


def _get_complete_results(db: Session,
                          labeler_id: Optional[int] = None,
                          region:     Optional[str]  = None,
                          sza_bin:    Optional[str]  = None,
                          verdict:    Optional[str]  = None):
    """Query all complete annotation results matching the given filters."""
    q = (
        db.query(AnnotationResult, Assignment, Chip)
        .join(Assignment, AnnotationResult.assignment_id == Assignment.id)
        .join(Chip,       Assignment.chip_id == Chip.id)
        .filter(Assignment.status == "complete")
    )
    if labeler_id:
        q = q.filter(Assignment.labeler_id == labeler_id)
    if region:
        q = q.filter(Chip.region == region)
    if sza_bin:
        q = q.filter(Chip.sza_bin == sza_bin)
    if verdict:
        q = q.filter(AnnotationResult.chip_verdict == verdict)
    return q.all()


# ── COCO JSON export ──────────────────────────────────────────────────────────

@router.get("/coco")
def export_coco(
    labeler_id: Optional[int] = Query(None),
    region:     Optional[str] = Query(None),
    sza_bin:    Optional[str] = Query(None),
    verdict:    Optional[str] = Query(None),
    db: Session = Depends(get_db),
    admin: Labeler = Depends(_require_admin),
):
    """
    Export accepted/edited annotations in COCO instance segmentation format.

    Only polygons with action='accepted', 'added', or 'modified' are included
    (rejected predictions are omitted from the output).
    """
    rows = _get_complete_results(db, labeler_id, region, sza_bin, verdict)

    coco = {
        "info": {
            "description":  "Iceberg labeling annotations",
            "date_created": datetime.utcnow().isoformat(),
            "version":      "1.0",
        },
        "licenses":    [],
        "categories": [
            {"id": 1, "name": "iceberg", "supercategory": "object"},
            {"id": 2, "name": "shadow",  "supercategory": "object"},
        ],
        "images":      [],
        "annotations": [],
    }

    cat_map   = {"iceberg": 1, "shadow": 2}
    image_map = {}   # chip_id → coco image id
    ann_id    = 1

    for result, assignment, chip in rows:
        # Register image (once per chip)
        if chip.id not in image_map:
            img_id          = len(coco["images"]) + 1
            image_map[chip.id] = img_id
            coco["images"].append({
                "id":          img_id,
                "file_name":   chip.filename,
                "width":       chip.width,
                "height":      chip.height,
                "region":      chip.region,
                "sza_bin":     chip.sza_bin,
            })

        img_id = image_map[chip.id]

        # Add polygon decisions (accepted/added/modified only)
        decisions = (
            db.query(PolygonDecision)
            .filter(PolygonDecision.result_id == result.id,
                    PolygonDecision.action.in_(["accepted", "added", "modified"]))
            .all()
        )
        for dec in decisions:
            coords   = json.loads(dec.pixel_coords)   # [[x,y], ...]
            flat     = [v for pt in coords for v in pt]
            xs       = [pt[0] for pt in coords]
            ys       = [pt[1] for pt in coords]
            bbox     = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
            coco["annotations"].append({
                "id":            ann_id,
                "image_id":      img_id,
                "category_id":   cat_map.get(dec.class_name, 1),
                "segmentation":  [flat],
                "area":          round(dec.area_px or 0.0, 2),
                "bbox":          [round(v, 2) for v in bbox],
                "iscrowd":       0,
                "action":        dec.action,
                "labeler_id":    assignment.labeler_id,
            })
            ann_id += 1

    content  = json.dumps(coco, indent=2)
    filename = f"iceberg_annotations_coco_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── CSV export ────────────────────────────────────────────────────────────────

@router.get("/csv")
def export_csv(
    labeler_id: Optional[int] = Query(None),
    region:     Optional[str] = Query(None),
    sza_bin:    Optional[str] = Query(None),
    db: Session = Depends(get_db),
    admin: Labeler = Depends(_require_admin),
):
    """
    Export a per-chip summary CSV with one row per annotation result.

    Columns: chip_filename, region, sza_bin, labeler_id, labeler_name,
             verdict, n_accepted, n_rejected, n_added, notes, submitted_at
    """
    rows = _get_complete_results(db, labeler_id, region, sza_bin)

    output  = io.StringIO()
    writer  = csv.writer(output)
    writer.writerow([
        "chip_id", "chip_filename", "region", "sza_bin",
        "labeler_id", "labeler_name", "verdict",
        "n_accepted", "n_rejected", "n_added", "n_modified",
        "prediction_count", "notes", "submitted_at",
    ])

    for result, assignment, chip in rows:
        labeler   = db.query(Labeler).filter(
            Labeler.id == assignment.labeler_id).first()
        decisions = db.query(PolygonDecision).filter(
            PolygonDecision.result_id == result.id).all()

        n_accepted = sum(1 for d in decisions if d.action == "accepted")
        n_rejected = sum(1 for d in decisions if d.action == "rejected")
        n_added    = sum(1 for d in decisions if d.action == "added")
        n_modified = sum(1 for d in decisions if d.action == "modified")

        writer.writerow([
            chip.id,
            chip.filename,
            chip.region or "",
            chip.sza_bin or "",
            assignment.labeler_id,
            labeler.name if labeler else "",
            result.chip_verdict,
            n_accepted,
            n_rejected,
            n_added,
            n_modified,
            chip.prediction_count,
            result.notes or "",
            result.submitted_at.isoformat(),
        ])

    output.seek(0)
    filename = f"iceberg_annotations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Summary stats ─────────────────────────────────────────────────────────────

@router.get("/summary")
def export_summary(
    db: Session = Depends(get_db),
    admin: Labeler = Depends(_require_admin),
):
    """Return high-level numbers for quick dashboard inspection."""
    total_complete = (
        db.query(Assignment).filter(Assignment.status == "complete").count()
    )
    verdicts = {}
    for result in db.query(AnnotationResult).all():
        v = result.chip_verdict
        verdicts[v] = verdicts.get(v, 0) + 1

    action_counts = {}
    for dec in db.query(PolygonDecision).all():
        action_counts[dec.action] = action_counts.get(dec.action, 0) + 1

    return {
        "total_complete_chips": total_complete,
        "verdicts":             verdicts,
        "polygon_actions":      action_counts,
    }
