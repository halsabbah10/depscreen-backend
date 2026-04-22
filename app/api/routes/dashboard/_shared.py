"""
Shared imports, constants, and helpers for the dashboard sub-modules.
"""

import logging

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.db import (
    User,
)
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

_rag_service = None


async def _get_rag(settings: Settings = Depends(get_settings)):
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
        await _rag_service.initialize()
    return _rag_service


def _verify_patient_access(db: Session, patient_id: str, clinician_id: str) -> User:
    """Return the patient or raise 404/403."""
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != clinician_id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")
    return patient
