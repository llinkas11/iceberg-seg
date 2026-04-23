"""
Authentication router.

Labelers register with their name and receive a UUID token.
They store this token locally (in the browser) and send it as a Bearer token
on every subsequent request.  There are no passwords — the token is the secret.

The admin token is separate and is set in config.py.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Labeler
from schemas import RegisterRequest, LoginRequest, AuthResponse
import config

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new labeler.  If the name is already taken, return 409.
    The generated token must be saved by the labeler — it cannot be recovered.
    """
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty")

    existing = db.query(Labeler).filter(Labeler.name == name).first()
    if existing:
        raise HTTPException(409, f"Name '{name}' is already registered. "
                                 "Ask the admin to reset your token if needed.")

    token      = str(uuid.uuid4())
    is_admin   = (token == config.ADMIN_TOKEN)   # extremely unlikely, but guard it
    group_name = (req.group_name or "").strip()
    labeler    = Labeler(name=name, token=token, group_name=group_name,
                         is_admin=is_admin)
    db.add(labeler)
    db.commit()
    db.refresh(labeler)
    return AuthResponse(labeler_id=labeler.id, name=labeler.name,
                        token=labeler.token, group_name=labeler.group_name or "",
                        is_admin=labeler.is_admin)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Log back in with name + token (e.g. after clearing browser storage).
    """
    labeler = db.query(Labeler).filter(
        Labeler.name  == req.name.strip(),
        Labeler.token == req.token.strip()
    ).first()
    if not labeler:
        raise HTTPException(401, "Name / token combination not found")
    return AuthResponse(labeler_id=labeler.id, name=labeler.name,
                        token=labeler.token, group_name=labeler.group_name or "",
                        is_admin=labeler.is_admin)


@router.post("/admin-login", response_model=AuthResponse)
def admin_login(token: str, db: Session = Depends(get_db)):
    """Log in using the admin token set in config.py."""
    if token != config.ADMIN_TOKEN:
        raise HTTPException(403, "Invalid admin token")

    admin = db.query(Labeler).filter(Labeler.is_admin == True).first()
    if not admin:
        # Auto-create the admin account on first use
        admin = Labeler(name="admin", token=config.ADMIN_TOKEN, is_admin=True)
        db.add(admin)
        db.commit()
        db.refresh(admin)
    return AuthResponse(labeler_id=admin.id, name=admin.name,
                        token=admin.token, group_name="",
                        is_admin=True)
