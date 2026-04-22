"""
Patient diagnoses and medications CRUD endpoints.
"""

from datetime import date, datetime
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.middleware.rate_limiter import limiter
from app.models.db import (
    Diagnosis,
    Medication,
    Notification,
    User,
    get_db,
)
from app.schemas.analysis import (
    DiagnosisCreate,
    DiagnosisResponse,
    MedicationCreate,
    MedicationResponse,
)
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()


def _diagnosis_to_response(dx: Diagnosis) -> DiagnosisResponse:
    return DiagnosisResponse(
        id=dx.id,
        condition=dx.condition,
        icd10_code=dx.icd10_code,
        diagnosed_date=dx.diagnosed_date.isoformat() if dx.diagnosed_date else None,
        status=dx.status,
        diagnosed_by=dx.diagnosed_by,
        notes=dx.notes,
        created_at=dx.created_at,
    )


def _medication_to_response(med: Medication) -> MedicationResponse:
    return MedicationResponse(
        id=med.id,
        name=med.name,
        dosage=med.dosage,
        frequency=med.frequency,
        start_date=med.start_date.isoformat() if med.start_date else None,
        end_date=med.end_date.isoformat() if med.end_date else None,
        prescribed_by=med.prescribed_by,
        notes=med.notes,
        is_active=med.is_active,
        created_at=med.created_at,
    )


# ── Diagnoses ─────────────────────────────────────────────────────────────────


@router.get("/patients/{patient_id}/diagnoses", response_model=list[DiagnosisResponse])
async def list_diagnoses(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List a patient's diagnoses (clinician view)."""
    _verify_patient_access(db, patient_id, current_user.id)
    rows = db.query(Diagnosis).filter(Diagnosis.patient_id == patient_id).order_by(desc(Diagnosis.created_at)).all()
    return [_diagnosis_to_response(dx) for dx in rows]


@router.post("/patients/{patient_id}/diagnoses", response_model=DiagnosisResponse, status_code=201)
@limiter.limit("30/minute")
async def add_diagnosis(
    patient_id: str,
    request: Request,
    payload: DiagnosisCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Add a diagnosis to a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    dx_id = str(uuid4())
    diagnosed_dt = None
    if payload.diagnosed_date:
        diagnosed_dt = date.fromisoformat(payload.diagnosed_date)

    dx = Diagnosis(
        id=dx_id,
        patient_id=patient_id,
        condition=payload.condition,
        icd10_code=payload.icd10_code,
        diagnosed_date=diagnosed_dt,
        status=payload.status,
        diagnosed_by=payload.diagnosed_by or current_user.full_name,
        notes=payload.notes,
    )
    db.add(dx)
    db.commit()
    db.refresh(dx)

    log_audit(db, current_user.id, "diagnosis_added", resource_type="diagnosis", resource_id=dx_id)
    return _diagnosis_to_response(dx)


@router.put("/diagnoses/{diagnosis_id}", response_model=DiagnosisResponse)
@limiter.limit("30/minute")
async def update_diagnosis(
    diagnosis_id: str,
    request: Request,
    status: str = Body(..., embed=True),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Update a diagnosis status (active, remission, resolved)."""
    dx = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not dx:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # Verify clinician has access to this patient
    _verify_patient_access(db, dx.patient_id, current_user.id)

    valid_statuses = {"active", "remission", "resolved"}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {', '.join(sorted(valid_statuses))}",
        )

    dx.status = status
    dx.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dx)

    log_audit(db, current_user.id, "diagnosis_updated", resource_type="diagnosis", resource_id=diagnosis_id)
    return _diagnosis_to_response(dx)


@router.delete("/diagnoses/{diagnosis_id}")
@limiter.limit("30/minute")
async def delete_diagnosis(
    diagnosis_id: str,
    request: Request,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Remove a diagnosis from a patient's record."""
    dx = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not dx:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    _verify_patient_access(db, dx.patient_id, current_user.id)

    db.delete(dx)
    db.commit()
    log_audit(db, current_user.id, "diagnosis_deleted", resource_type="diagnosis", resource_id=diagnosis_id)
    return {"status": "deleted", "diagnosis_id": diagnosis_id}


# ── Medications ───────────────────────────────────────────────────────────────


@router.get("/patients/{patient_id}/medications", response_model=list[MedicationResponse])
async def list_patient_medications(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List a patient's medications (clinician view)."""
    _verify_patient_access(db, patient_id, current_user.id)
    meds = db.query(Medication).filter(Medication.patient_id == patient_id).order_by(desc(Medication.created_at)).all()
    return [_medication_to_response(m) for m in meds]


@router.post("/patients/{patient_id}/medications", response_model=MedicationResponse, status_code=201)
@limiter.limit("30/minute")
async def add_patient_medication(
    patient_id: str,
    request: Request,
    payload: MedicationCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician adds a medication to a patient's record."""
    _verify_patient_access(db, patient_id, current_user.id)

    start_dt = None
    if payload.start_date:
        try:
            start_dt = date.fromisoformat(payload.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")

    end_dt = None
    if payload.end_date:
        try:
            end_dt = date.fromisoformat(payload.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")

    if start_dt and end_dt and end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date cannot be before start_date.")

    med_id = str(uuid4())
    med = Medication(
        id=med_id,
        patient_id=patient_id,
        name=payload.name,
        dosage=payload.dosage,
        frequency=payload.frequency,
        start_date=start_dt,
        end_date=end_dt,
        prescribed_by=payload.prescribed_by or current_user.full_name,
        notes=payload.notes,
        is_active=True,
    )
    db.add(med)
    db.commit()
    db.refresh(med)

    log_audit(db, current_user.id, "medication_added_by_clinician", resource_type="medication", resource_id=med_id)

    # Notify patient that a clinician added a medication
    db.add(
        Notification(
            user_id=patient_id,
            notification_type="care_plan_updated",
            title="Your clinician updated your medications",
            message=f"{current_user.full_name} added {med.name} to your record. No action needed from you right now — this is just so you have visibility.",
            link="/profile",
        )
    )
    db.commit()

    return _medication_to_response(med)


@router.put("/medications/{medication_id}", response_model=MedicationResponse)
@limiter.limit("30/minute")
async def update_patient_medication(
    medication_id: str,
    request: Request,
    payload: MedicationCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician updates an existing medication on a patient's record."""
    med = db.query(Medication).filter(Medication.id == medication_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")

    _verify_patient_access(db, med.patient_id, current_user.id)

    med.name = payload.name
    med.dosage = payload.dosage
    med.frequency = payload.frequency
    med.prescribed_by = payload.prescribed_by
    med.notes = payload.notes

    if payload.start_date:
        try:
            med.start_date = date.fromisoformat(payload.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    else:
        med.start_date = None

    if payload.end_date:
        try:
            med.end_date = date.fromisoformat(payload.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
    else:
        med.end_date = None

    if med.start_date and med.end_date and med.end_date < med.start_date:
        raise HTTPException(status_code=400, detail="end_date cannot be before start_date.")

    db.commit()
    db.refresh(med)
    log_audit(
        db, current_user.id, "medication_updated_by_clinician", resource_type="medication", resource_id=medication_id
    )

    return _medication_to_response(med)


@router.delete("/medications/{medication_id}")
@limiter.limit("30/minute")
async def deactivate_patient_medication(
    medication_id: str,
    request: Request,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician deactivates (soft-deletes) a medication."""
    med = db.query(Medication).filter(Medication.id == medication_id).first()
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")

    _verify_patient_access(db, med.patient_id, current_user.id)

    med.is_active = False
    db.commit()
    log_audit(
        db,
        current_user.id,
        "medication_deactivated_by_clinician",
        resource_type="medication",
        resource_id=medication_id,
    )

    return {"status": "deactivated", "medication_id": medication_id}
