"""Pydantic request/response schemas."""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    group_name: str = ""

class LoginRequest(BaseModel):
    name: str
    token: str

class AuthResponse(BaseModel):
    labeler_id: int
    name: str
    token: str
    group_name: str
    is_admin: bool


# ── Chips & Predictions ───────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    id: int
    class_name: str
    pixel_coords: list[list[float]]   # [[col, row], ...]
    area_m2: Optional[float]

    model_config = {"from_attributes": True}

class ChipOut(BaseModel):
    id: int
    filename: str
    region: Optional[str]
    sza_bin: Optional[str]
    width: int
    height: int
    prediction_count: int

    model_config = {"from_attributes": True}

class ChipWithPredictions(ChipOut):
    predictions: list[PredictionOut]
    assignment_id: int
    assignment_status: str


# ── Annotations ───────────────────────────────────────────────────────────────

class PolygonDecisionIn(BaseModel):
    prediction_id: Optional[int] = None   # None → labeler-drawn polygon
    action: str                           # "accepted"|"rejected"|"added"|"modified"
    class_name: str
    pixel_coords: list[list[float]]       # [[col, row], ...]

class AnnotationSubmit(BaseModel):
    chip_id: int
    assignment_id: int
    verdict: str                          # "accepted"|"rejected"|"edited"|"skipped"
    polygon_decisions: list[PolygonDecisionIn]
    notes: str = ""
    tags: list[str] = []                  # ["sea-ice", "ambiguous", "cloud", …]

class AnnotationResponse(BaseModel):
    annotation_result_id: int
    next_chip_available: bool


# ── Admin ─────────────────────────────────────────────────────────────────────

class LabelerProgress(BaseModel):
    id: int
    name: str
    group_name: str
    is_admin: bool
    total_assigned: int
    complete: int
    pending: int
    skipped: int
    created_at: datetime

class AdminStats(BaseModel):
    total_chips: int
    total_assigned: int
    total_complete: int
    total_pending: int
    labeler_count: int

class AssignRequest(BaseModel):
    labeler_id: int
    num_chips: int
    region: Optional[str] = None
    sza_bin: Optional[str] = None
    exclude_complete: bool = True   # don't re-assign already-complete chips

class AssignResponse(BaseModel):
    assigned: int
    labeler_name: str

class ChipStatusOut(BaseModel):
    id: int
    filename: str
    region: Optional[str]
    sza_bin: Optional[str]
    prediction_count: int
    assignments: list[dict]

    model_config = {"from_attributes": True}


# ── Export ────────────────────────────────────────────────────────────────────

class ExportFilters(BaseModel):
    labeler_id: Optional[int] = None
    region: Optional[str] = None
    sza_bin: Optional[str] = None
    verdict: Optional[str] = None   # filter to chips with this verdict


# ── SAM Segmentation ────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    chip_id: int
    point_coords: list[list[float]]   # [[col, row], ...]  pixel space
    point_labels: list[int]            # 1=foreground, 0=background
    multimask: bool = False            # return best mask only by default

class SegmentResponse(BaseModel):
    polygon: list[list[float]]         # [[col, row], ...]  or [] if no mask
    confidence: float                  # 0.0–1.0
    area_px: float                     # polygon area in pixels
