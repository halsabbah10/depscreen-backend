"""
Shared imports, constants, and helpers for the dashboard sub-modules.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.db import (
    User,
)
from app.services.container import get_rag_service
from app.services.rag import RAGService

logger = logging.getLogger(__name__)


def _get_rag() -> RAGService | None:
    return get_rag_service()


def _verify_patient_access(db: Session, patient_id: str, clinician_id: str) -> User:
    """Return the patient or raise 404/403."""
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != clinician_id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")
    return patient
