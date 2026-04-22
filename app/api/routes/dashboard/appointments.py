"""
Appointment CRUD endpoints.
"""

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import (
    Appointment,
    Notification,
    User,
    get_db,
)
from app.schemas.analysis import (
    AppointmentCreate,
    AppointmentResponse,
    AppointmentStatusUpdate,
)
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()

# Valid status transitions for appointments
_APPOINTMENT_TRANSITIONS: dict[str, set[str]] = {
    "scheduled": {"confirmed", "cancelled"},
    "confirmed": {"completed", "cancelled", "no_show"},
    # terminal states — no outgoing edges
    "completed": set(),
    "cancelled": set(),
    "no_show": set(),
}


def _appointment_to_response(appt: Appointment, db: Session | None = None) -> AppointmentResponse:
    patient_name = None
    clinician_name = None
    if db is not None:
        if appt.patient_id:
            p = db.query(User).filter(User.id == appt.patient_id).first()
            patient_name = p.full_name if p else None
        if appt.clinician_id:
            c = db.query(User).filter(User.id == appt.clinician_id).first()
            clinician_name = c.full_name if c else None
    return AppointmentResponse(
        id=appt.id,
        patient_id=appt.patient_id,
        patient_name=patient_name,
        clinician_id=appt.clinician_id,
        clinician_name=clinician_name,
        scheduled_at=appt.scheduled_at,
        duration_minutes=appt.duration_minutes,
        appointment_type=appt.appointment_type,
        status=appt.status,
        notes=appt.notes,
        location=appt.location,
        created_at=appt.created_at,
    )


@router.get("/appointments", response_model=list[AppointmentResponse])
async def list_appointments(
    status: str = Query(None, description="Filter by status, or 'all' for every status"),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List the clinician's appointments. By default returns upcoming scheduled/confirmed only."""
    query = db.query(Appointment).filter(Appointment.clinician_id == current_user.id)

    if status and status != "all":
        query = query.filter(Appointment.status == status)
    elif not status:
        # Default: upcoming non-terminal appointments
        query = query.filter(
            Appointment.status.in_(["scheduled", "confirmed"]),
            Appointment.scheduled_at >= datetime.utcnow(),
        )

    appointments = query.order_by(Appointment.scheduled_at).all()
    return [_appointment_to_response(a, db) for a in appointments]


@router.post("/appointments", response_model=AppointmentResponse, status_code=201)
@limiter.limit("30/minute")
async def create_appointment(
    request: Request,
    payload: AppointmentCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Create a new appointment for a patient."""
    _verify_patient_access(db, payload.patient_id, current_user.id)

    appt_id = str(uuid4())
    appt = Appointment(
        id=appt_id,
        patient_id=payload.patient_id,
        clinician_id=current_user.id,
        scheduled_at=datetime.fromisoformat(payload.scheduled_at),
        duration_minutes=payload.duration_minutes,
        appointment_type=payload.appointment_type,
        status="scheduled",
        notes=payload.notes,
        location=payload.location,
    )
    db.add(appt)

    # Also create an in-app notification for the patient
    patient = db.query(User).filter(User.id == payload.patient_id).first()
    formatted = appt.scheduled_at.strftime("%d/%m/%Y at %H:%M")
    db.add(
        Notification(
            user_id=payload.patient_id,
            notification_type="appointment_scheduled",
            title="New appointment scheduled",
            message=f"Your clinician scheduled an appointment on {formatted}.",
            link="/appointments",
        )
    )

    db.commit()
    db.refresh(appt)

    # Email the patient immediately (confirmation). The 24h reminder fires separately via scheduler.
    try:
        from app.services.email import get_email_service

        if patient and patient.email and patient.email_notifications:
            get_email_service(settings).send_appointment_reminder(
                patient_name=patient.full_name,
                patient_email=patient.email,
                appointment_at=formatted,
                clinician_name=current_user.full_name,
            )
    except Exception as e:
        import logging as _logging

        _logging.getLogger(__name__).warning(f"Appointment email failed: {e}")

    log_audit(db, current_user.id, "appointment_created", resource_type="appointment", resource_id=appt_id)
    return _appointment_to_response(appt, db)


@router.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def get_appointment(
    appointment_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get a single appointment's details."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")
    return _appointment_to_response(appt, db)


@router.patch("/appointments/{appointment_id}/status", response_model=AppointmentResponse)
@limiter.limit("30/minute")
async def update_appointment_status(
    appointment_id: str,
    request: Request,
    payload: AppointmentStatusUpdate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Transition an appointment's status with validation."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")

    allowed = _APPOINTMENT_TRANSITIONS.get(appt.status, set())
    if not allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Appointment is in terminal state '{appt.status}' and cannot be changed",
        )
    if payload.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot transition from '{appt.status}' to '{payload.status}'. Allowed: {', '.join(sorted(allowed))}",
        )

    appt.status = payload.status
    appt.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(appt)

    log_audit(
        db, current_user.id, f"appointment_{payload.status}", resource_type="appointment", resource_id=appointment_id
    )
    return _appointment_to_response(appt, db)


@router.delete("/appointments/{appointment_id}", response_model=AppointmentResponse)
@limiter.limit("30/minute")
async def cancel_appointment(
    appointment_id: str,
    request: Request,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Cancel an appointment (shorthand for setting status to 'cancelled')."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")

    if appt.status in ("completed", "cancelled", "no_show"):
        raise HTTPException(
            status_code=400,
            detail=f"Appointment is already in terminal state '{appt.status}'",
        )

    appt.status = "cancelled"
    appt.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(appt)

    log_audit(db, current_user.id, "appointment_cancelled", resource_type="appointment", resource_id=appointment_id)
    return _appointment_to_response(appt, db)
