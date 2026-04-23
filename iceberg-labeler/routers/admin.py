"""
Admin router — requires admin token.

All endpoints here are protected by the admin token set in config.py.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models import Labeler, Chip, Assignment
from schemas import (AdminStats, LabelerProgress, AssignRequest, AssignResponse,
                     ChipStatusOut)
import config

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _require_admin(authorization: str = Header(...),
                   db: Session = Depends(get_db)) -> Labeler:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token == config.ADMIN_TOKEN:
        # Token matches config — return or create the admin labeler
        admin = db.query(Labeler).filter(Labeler.is_admin == True).first()
        if not admin:
            admin = Labeler(name="admin", token=config.ADMIN_TOKEN, is_admin=True)
            db.add(admin)
            db.commit()
            db.refresh(admin)
        return admin
    # Also accept a registered admin labeler token
    labeler = db.query(Labeler).filter(Labeler.token == token,
                                       Labeler.is_admin == True).first()
    if not labeler:
        raise HTTPException(403, "Admin access required")
    return labeler


# ── Stats ─────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=AdminStats)
def get_stats(db: Session = Depends(get_db),
              admin: Labeler = Depends(_require_admin)):
    total_chips    = db.query(Chip).count()
    total_assigned = db.query(Assignment).count()
    total_complete = db.query(Assignment).filter(
        Assignment.status == "complete").count()
    total_pending  = db.query(Assignment).filter(
        Assignment.status.in_(["pending", "in_progress"])).count()
    labeler_count  = db.query(Labeler).filter(
        Labeler.is_admin == False).count()
    return AdminStats(
        total_chips    = total_chips,
        total_assigned = total_assigned,
        total_complete = total_complete,
        total_pending  = total_pending,
        labeler_count  = labeler_count,
    )


@router.get("/labelers", response_model=list[LabelerProgress])
def get_labelers(db: Session = Depends(get_db),
                 admin: Labeler = Depends(_require_admin)):
    labelers = db.query(Labeler).filter(Labeler.is_admin == False).all()
    result   = []
    for labeler in labelers:
        assignments = db.query(Assignment).filter(
            Assignment.labeler_id == labeler.id).all()
        total    = len(assignments)
        complete = sum(1 for a in assignments if a.status == "complete")
        skipped  = sum(1 for a in assignments if a.status == "skipped")
        pending  = total - complete - skipped
        result.append(LabelerProgress(
            id             = labeler.id,
            name           = labeler.name,
            group_name     = labeler.group_name or "",
            is_admin       = labeler.is_admin,
            total_assigned = total,
            complete       = complete,
            pending        = pending,
            skipped        = skipped,
            created_at     = labeler.created_at,
        ))
    return result


# ── Chip browser ──────────────────────────────────────────────────────────────

@router.get("/chips")
def get_chips(region: str | None = None,
              sza_bin: str | None = None,
              status: str | None = None,
              limit: int = 200,
              offset: int = 0,
              db: Session = Depends(get_db),
              admin: Labeler = Depends(_require_admin)):
    """List chips with their assignment status. Filterable by region/sza_bin."""
    q = db.query(Chip)
    if region:
        q = q.filter(Chip.region == region)
    if sza_bin:
        q = q.filter(Chip.sza_bin == sza_bin)
    total  = q.count()
    chips  = q.offset(offset).limit(limit).all()

    rows = []
    for chip in chips:
        asgns = db.query(Assignment).filter(Assignment.chip_id == chip.id).all()
        rows.append({
            "id":               chip.id,
            "filename":         chip.filename,
            "region":           chip.region,
            "sza_bin":          chip.sza_bin,
            "prediction_count": chip.prediction_count,
            "assignments": [
                {
                    "labeler_id":   a.labeler_id,
                    "status":       a.status,
                    "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                }
                for a in asgns
            ],
        })
    return {"total": total, "chips": rows}


@router.get("/regions")
def get_regions(db: Session = Depends(get_db),
                admin: Labeler = Depends(_require_admin)):
    """Return distinct region and sza_bin values present in the database."""
    regions  = [r[0] for r in db.query(Chip.region).distinct().all() if r[0]]
    sza_bins = [r[0] for r in db.query(Chip.sza_bin).distinct().all() if r[0]]
    return {"regions": sorted(regions), "sza_bins": sorted(sza_bins)}


# ── Assignment ────────────────────────────────────────────────────────────────

@router.post("/assign", response_model=AssignResponse)
def assign_chips(req: AssignRequest,
                 db: Session = Depends(get_db),
                 admin: Labeler = Depends(_require_admin)):
    """
    Assign `num_chips` chips to labeler `labeler_id`.

    Selects chips that:
      - match region/sza_bin filters (if given)
      - have no existing complete assignment for this labeler (when exclude_complete)
      - prioritise chips that have not been assigned to anyone yet
    """
    labeler = db.query(Labeler).filter(Labeler.id == req.labeler_id).first()
    if not labeler:
        raise HTTPException(404, f"Labeler {req.labeler_id} not found")

    # Chips already assigned to this labeler
    already = {a.chip_id for a in db.query(Assignment).filter(
        Assignment.labeler_id == labeler.id).all()}

    q = db.query(Chip)
    if req.region:
        q = q.filter(Chip.region == req.region)
    if req.sza_bin:
        q = q.filter(Chip.sza_bin == req.sza_bin)

    # All matching chips not yet assigned to this labeler
    candidates = [c for c in q.all() if c.id not in already]

    # Sort: unassigned chips first, then least-assigned
    def _priority(chip):
        count = db.query(Assignment).filter(Assignment.chip_id == chip.id).count()
        return count

    candidates.sort(key=_priority)
    to_assign  = candidates[:req.num_chips]

    for chip in to_assign:
        asgn = Assignment(chip_id=chip.id, labeler_id=labeler.id)
        db.add(asgn)
    db.commit()

    return AssignResponse(assigned=len(to_assign), labeler_name=labeler.name)


@router.delete("/assignment/{assignment_id}")
def delete_assignment(assignment_id: int,
                      db: Session = Depends(get_db),
                      admin: Labeler = Depends(_require_admin)):
    """Remove an assignment (e.g. to re-assign it)."""
    asgn = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    if not asgn:
        raise HTTPException(404)
    if asgn.result:
        raise HTTPException(409, "Cannot delete an assignment that has been submitted")
    db.delete(asgn)
    db.commit()
    return {"deleted": assignment_id}


@router.post("/assign-all-unassigned")
def assign_all_unassigned(labeler_id: int,
                          db: Session = Depends(get_db),
                          admin: Labeler = Depends(_require_admin)):
    """Assign every chip that has no assignment to the given labeler."""
    labeler = db.query(Labeler).filter(Labeler.id == labeler_id).first()
    if not labeler:
        raise HTTPException(404)

    assigned_chip_ids = {a.chip_id for a in db.query(Assignment).all()}
    unassigned = db.query(Chip).filter(~Chip.id.in_(assigned_chip_ids)).all()
    for chip in unassigned:
        db.add(Assignment(chip_id=chip.id, labeler_id=labeler.id))
    db.commit()
    return {"assigned": len(unassigned), "labeler_name": labeler.name}


# ── Labeler management ────────────────────────────────────────────────────────

@router.delete("/labeler/{labeler_id}")
def delete_labeler(labeler_id: int,
                   db: Session = Depends(get_db),
                   admin: Labeler = Depends(_require_admin)):
    """Delete a labeler and all their assignments (use with care)."""
    labeler = db.query(Labeler).filter(Labeler.id == labeler_id).first()
    if not labeler:
        raise HTTPException(404)
    if labeler.is_admin:
        raise HTTPException(400, "Cannot delete admin labeler")
    db.delete(labeler)
    db.commit()
    return {"deleted": labeler_id}
