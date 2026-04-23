"""SQLAlchemy ORM models for iceberg-labeler."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text,
    DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship

from database import Base


class Chip(Base):
    """A 256×256 satellite image chip to be labeled."""
    __tablename__ = "chips"

    id             = Column(Integer, primary_key=True, index=True)
    filename       = Column(String, unique=True, nullable=False, index=True)
    image_path     = Column(String, nullable=False)   # absolute path to PNG
    tif_path       = Column(String)                   # absolute path to source TIF
    region         = Column(String, index=True)       # "kq" | "sk" | …
    sza_bin        = Column(String, index=True)       # "sza_lt65" | "sza_65_70" | …
    width          = Column(Integer, default=256)
    height         = Column(Integer, default=256)
    prediction_count = Column(Integer, default=0)     # cached count of predictions
    created_at     = Column(DateTime, default=datetime.utcnow)

    predictions    = relationship("Prediction", back_populates="chip",
                                  cascade="all, delete-orphan")
    assignments    = relationship("Assignment", back_populates="chip",
                                  cascade="all, delete-orphan")


class Prediction(Base):
    """A single polygon predicted by UNet++ for one chip."""
    __tablename__ = "predictions"

    id           = Column(Integer, primary_key=True, index=True)
    chip_id      = Column(Integer, ForeignKey("chips.id"), nullable=False, index=True)
    class_name   = Column(String, nullable=False)   # "iceberg" | "shadow"
    # Pixel-space polygon stored as JSON: [[col, row], [col, row], ...]
    # (0,0) = top-left; col increases right; row increases down.
    pixel_coords = Column(Text, nullable=False)
    area_m2      = Column(Float)

    chip         = relationship("Chip", back_populates="predictions")
    decisions    = relationship("PolygonDecision", back_populates="prediction")


class Labeler(Base):
    """A human labeler (student / image tester)."""
    __tablename__ = "labelers"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, unique=True, nullable=False, index=True)
    token      = Column(String, unique=True, nullable=False, index=True)
    group_name = Column(String, default="")            # "Group1" | "Group2" | …
    is_admin   = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    assignments = relationship("Assignment", back_populates="labeler")


class Assignment(Base):
    """One chip assigned to one labeler."""
    __tablename__ = "assignments"
    __table_args__ = (
        UniqueConstraint("chip_id", "labeler_id", name="uq_chip_labeler"),
    )

    id           = Column(Integer, primary_key=True, index=True)
    chip_id      = Column(Integer, ForeignKey("chips.id"), nullable=False, index=True)
    labeler_id   = Column(Integer, ForeignKey("labelers.id"), nullable=False, index=True)
    # pending → in_progress → complete | skipped
    status       = Column(String, default="pending", index=True)
    assigned_at  = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    chip    = relationship("Chip", back_populates="assignments")
    labeler = relationship("Labeler", back_populates="assignments")
    result  = relationship("AnnotationResult", back_populates="assignment",
                           uselist=False, cascade="all, delete-orphan")


class AnnotationResult(Base):
    """The high-level verdict a labeler submitted for one chip."""
    __tablename__ = "annotation_results"

    id           = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"),
                           nullable=False, unique=True, index=True)
    # "accepted" = all predictions correct
    # "rejected" = chip contains no icebergs (all predictions wrong)
    # "edited"   = labeler reviewed individual polygons
    # "skipped"  = labeler deferred the chip
    chip_verdict = Column(String, nullable=False)
    notes        = Column(Text, default="")
    # Comma-separated tags: "sea-ice,ambiguous,cloud,land-edge"
    tags         = Column(String, default="")
    submitted_at = Column(DateTime, default=datetime.utcnow)

    assignment       = relationship("Assignment", back_populates="result")
    polygon_decisions = relationship("PolygonDecision", back_populates="result",
                                     cascade="all, delete-orphan")


class PolygonDecision(Base):
    """
    The per-polygon outcome within an edited annotation.
    For 'accepted'/'rejected' chip verdicts the individual decisions are
    auto-generated from the predictions; for 'edited' they are explicit.
    """
    __tablename__ = "polygon_decisions"

    id            = Column(Integer, primary_key=True, index=True)
    result_id     = Column(Integer, ForeignKey("annotation_results.id"),
                           nullable=False, index=True)
    # NULL when the polygon was added by the labeler (not from UNet++)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    # "accepted" | "rejected" | "added" | "modified"
    action        = Column(String, nullable=False)
    class_name    = Column(String, nullable=False)
    # Final pixel-space polygon: [[col, row], ...]
    pixel_coords  = Column(Text, nullable=False)
    area_px       = Column(Float)

    result     = relationship("AnnotationResult", back_populates="polygon_decisions")
    prediction = relationship("Prediction", back_populates="decisions")
