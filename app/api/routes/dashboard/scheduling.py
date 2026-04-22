"""
Patient screening schedule CRUD endpoints.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.middleware.rate_limiter import limiter
from app.models.db import (
    Notification,
    ScreeningSchedule,
    User,
    get_db,
)
from app.schemas.analysis import (
    ScreeningScheduleCreate,
    ScreeningScheduleResponse,
)
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()


def _schedule_to_response(schedule: ScreeningSchedule, db: Session) -> ScreeningScheduleResponse:
    assigned_name = None
    if schedule.assigned_by:
        clinician = db.query(User).filter(User.id == schedule.assigned_by).first()
        if clinician:
            assigned_name = clinician.full_name
    return ScreeningScheduleResponse(
        id=schedule.id,
        frequency=schedule.frequency,
        custom_days=schedule.custom_days,
        day_of_week=schedule.day_of_week,
        preferred_time=schedule.preferred_time,
        next_due_at=schedule.next_due_at,
        last_completed_at=schedule.last_completed_at,
        is_active=schedule.is_active,
        assigned_by=schedule.assigned_by,
        assigned_by_name=assigned_name,
        created_at=schedule.created_at,
    )


@router.get("/patients/{patient_id}/screening-schedule", response_model=ScreeningScheduleResponse | None)
async def get_patient_screening_schedule(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get a patient's active screening schedule (clinician view)."""
    _verify_patient_access(db, patient_id, current_user.id)
    schedule = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.patient_id == patient_id,
            ScreeningSchedule.is_active == True,
        )
        .first()
    )
    if not schedule:
        return None
    return _schedule_to_response(schedule, db)


@router.post("/patients/{patient_id}/screening-schedule", response_model=ScreeningScheduleResponse, status_code=201)
@limiter.limit("30/minute")
async def assign_patient_screening_schedule(
    patient_id: str,
    request: Request,
    body: ScreeningScheduleCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician assigns or updates a screening schedule for a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    valid_frequencies = {"weekly", "biweekly", "monthly", "custom"}
    if body.frequency not in valid_frequencies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frequency. Must be one of: {', '.join(sorted(valid_frequencies))}",
        )
    if body.frequency == "custom" and not body.custom_days:
        raise HTTPException(status_code=400, detail="custom_days is required when frequency is 'custom'.")
    if body.frequency in {"weekly", "biweekly"} and body.day_of_week is None:
        raise HTTPException(status_code=400, detail="day_of_week is required for weekly/biweekly schedules.")

    # Deactivate existing active schedules
    existing = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.patient_id == patient_id,
            ScreeningSchedule.is_active == True,
        )
        .all()
    )
    for s in existing:
        s.is_active = False

    now = datetime.utcnow()
    if body.frequency == "weekly":
        days_ahead = (body.day_of_week - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_due = now + timedelta(days=days_ahead)
    elif body.frequency == "biweekly":
        days_ahead = (body.day_of_week - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 14
        next_due = now + timedelta(days=days_ahead)
    elif body.frequency == "monthly":
        next_due = now + timedelta(days=30)
    elif body.frequency == "custom":
        next_due = now + timedelta(days=body.custom_days)
    else:
        next_due = now + timedelta(days=7)

    if body.preferred_time:
        try:
            hour, minute = map(int, body.preferred_time.split(":"))
            next_due = next_due.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid preferred_time format. Use HH:MM.")

    schedule_id = str(uuid4())
    schedule = ScreeningSchedule(
        id=schedule_id,
        patient_id=patient_id,
        frequency=body.frequency,
        custom_days=body.custom_days,
        day_of_week=body.day_of_week,
        preferred_time=body.preferred_time,
        next_due_at=next_due,
        is_active=True,
        assigned_by=current_user.id,
    )
    db.add(schedule)

    # Notify the patient
    freq_label = {
        "weekly": "weekly",
        "biweekly": "every two weeks",
        "monthly": "monthly",
        "custom": f"every {body.custom_days} days",
    }.get(body.frequency, body.frequency)
    db.add(
        Notification(
            user_id=patient_id,
            notification_type="care_plan_updated",
            title="Your clinician set up a check-in rhythm",
            message=f"{current_user.full_name} suggests a {freq_label} check-in. Whenever you're ready, your next one is waiting.",
            link="/screening",
        )
    )

    db.commit()
    db.refresh(schedule)

    log_audit(
        db,
        current_user.id,
        "screening_schedule_assigned_by_clinician",
        resource_type="screening_schedule",
        resource_id=schedule_id,
    )

    return _schedule_to_response(schedule, db)


@router.delete("/patients/{patient_id}/screening-schedule")
@limiter.limit("30/minute")
async def deactivate_patient_screening_schedule(
    patient_id: str,
    request: Request,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician deactivates a patient's active schedule."""
    _verify_patient_access(db, patient_id, current_user.id)
    active = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.patient_id == patient_id,
            ScreeningSchedule.is_active == True,
        )
        .all()
    )
    if not active:
        raise HTTPException(status_code=404, detail="No active schedule for this patient")

    for s in active:
        s.is_active = False

    db.commit()
    log_audit(db, current_user.id, "screening_schedule_deactivated_by_clinician", resource_type="screening_schedule")
    return {"status": "deactivated", "count": len(active)}
