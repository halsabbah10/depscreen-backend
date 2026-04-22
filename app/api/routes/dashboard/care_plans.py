"""
Care plan CRUD and care-plan-templates endpoints.
"""

from datetime import date, datetime
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import (
    CarePlan,
    Notification,
    User,
    get_db,
)
from app.schemas.analysis import (
    CarePlanCreate,
    CarePlanResponse,
)
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()

_CARE_PLAN_TEMPLATES = [
    {
        "name": "Behavioral Activation — Mild Depression",
        "goals": [
            {"text": "Schedule 3 pleasurable activities per week"},
            {"text": "Track daily mood on 1-10 scale"},
        ],
        "interventions": [
            {"name": "Activity scheduling", "frequency": "daily"},
            {"name": "Mood journaling", "frequency": "daily"},
        ],
    },
    {
        "name": "CBT-Based — Moderate Depression",
        "goals": [
            {"text": "Identify and challenge 2 negative thought patterns per week"},
            {"text": "Complete CBT thought records"},
        ],
        "interventions": [
            {"name": "Cognitive restructuring worksheet", "frequency": "twice_weekly"},
            {"name": "Behavioral experiments", "frequency": "weekly"},
        ],
    },
    {
        "name": "Crisis Stabilization — High Risk",
        "goals": [
            {"text": "Complete safety plan with clinician"},
            {"text": "Establish daily check-in routine"},
            {"text": "Connect with support network"},
        ],
        "interventions": [
            {"name": "Safety planning", "frequency": "immediate"},
            {"name": "Daily check-in screening", "frequency": "daily"},
            {"name": "Emergency contact verification", "frequency": "immediate"},
        ],
    },
    {
        "name": "Medication Management + Therapy",
        "goals": [
            {"text": "Medication adherence for 30 consecutive days"},
            {"text": "Attend weekly therapy sessions"},
        ],
        "interventions": [
            {"name": "Medication tracking", "frequency": "daily"},
            {"name": "Therapy session attendance", "frequency": "weekly"},
            {"name": "Side effect monitoring", "frequency": "weekly"},
        ],
    },
]


def _care_plan_to_response(cp: CarePlan, db: Session | None = None) -> CarePlanResponse:
    patient_name = None
    clinician_name = None
    if db is not None:
        if cp.patient_id:
            p = db.query(User).filter(User.id == cp.patient_id).first()
            patient_name = p.full_name if p else None
        if cp.clinician_id:
            c = db.query(User).filter(User.id == cp.clinician_id).first()
            clinician_name = c.full_name if c else None
    return CarePlanResponse(
        id=cp.id,
        patient_id=cp.patient_id,
        patient_name=patient_name,
        clinician_id=cp.clinician_id,
        clinician_name=clinician_name,
        title=cp.title,
        description=cp.description,
        template_name=cp.template_name,
        goals=cp.goals or [],
        interventions=cp.interventions or [],
        # ISO 8601 — frontend formats for display via formatDate()
        review_date=cp.review_date.isoformat() if cp.review_date else None,
        status=cp.status,
        created_at=cp.created_at,
        updated_at=cp.updated_at,
    )


@router.get("/patients/{patient_id}/care-plans", response_model=list[CarePlanResponse])
async def list_care_plans(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List all care plans for a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    plans = db.query(CarePlan).filter(CarePlan.patient_id == patient_id).order_by(desc(CarePlan.created_at)).all()
    return [_care_plan_to_response(cp, db) for cp in plans]


@router.post("/care-plans", response_model=CarePlanResponse, status_code=201)
@limiter.limit("30/minute")
async def create_care_plan(
    request: Request,
    payload: CarePlanCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Create a new care plan."""
    _verify_patient_access(db, payload.patient_id, current_user.id)

    cp_id = str(uuid4())
    review_dt = None
    if payload.review_date:
        review_dt = date.fromisoformat(payload.review_date)

    cp = CarePlan(
        id=cp_id,
        patient_id=payload.patient_id,
        clinician_id=current_user.id,
        title=payload.title,
        description=payload.description,
        template_name=payload.template_name,
        goals=[g.model_dump() for g in payload.goals],
        interventions=[i.model_dump() for i in payload.interventions],
        review_date=review_dt,
        status="active",
    )
    db.add(cp)
    db.commit()
    db.refresh(cp)

    log_audit(db, current_user.id, "care_plan_created", resource_type="care_plan", resource_id=cp_id)
    return _care_plan_to_response(cp, db)


@router.get("/care-plans/{care_plan_id}", response_model=CarePlanResponse)
async def get_care_plan(
    care_plan_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get a single care plan's details."""
    cp = db.query(CarePlan).filter(CarePlan.id == care_plan_id).first()
    if not cp:
        raise HTTPException(status_code=404, detail="Care plan not found")
    if cp.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This care plan is not assigned to you")
    return _care_plan_to_response(cp, db)


@router.put("/care-plans/{care_plan_id}", response_model=CarePlanResponse)
@limiter.limit("30/minute")
async def update_care_plan(
    care_plan_id: str,
    request: Request,
    goals: list[dict] = Body(None),
    interventions: list[dict] = Body(None),
    status: str = Body(None),
    review_date: str = Body(None),
    description: str = Body(None),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Update goals, interventions, status, or review date of a care plan."""
    cp = db.query(CarePlan).filter(CarePlan.id == care_plan_id).first()
    if not cp:
        raise HTTPException(status_code=404, detail="Care plan not found")
    if cp.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This care plan is not assigned to you")

    valid_statuses = {"active", "review_needed", "completed", "archived"}
    if status is not None:
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Status must be one of: {', '.join(sorted(valid_statuses))}",
            )
        cp.status = status

    if goals is not None:
        cp.goals = goals
    if interventions is not None:
        cp.interventions = interventions
    if review_date is not None:
        cp.review_date = date.fromisoformat(review_date)
    if description is not None:
        cp.description = description

    cp.updated_at = datetime.utcnow()

    # In-app notification for the patient
    db.add(
        Notification(
            user_id=cp.patient_id,
            notification_type="care_plan_updated",
            title="Your care plan has been updated",
            message=f"Your clinician updated: {cp.title}",
            link="/care-plan",
        )
    )

    db.commit()
    db.refresh(cp)

    # Email the patient
    try:
        from app.services.email import get_email_service

        patient = db.query(User).filter(User.id == cp.patient_id).first()
        if patient and patient.email and patient.email_notifications:
            get_email_service(settings).send_care_plan_update(
                patient_name=patient.full_name,
                patient_email=patient.email,
                plan_title=cp.title,
            )
    except Exception as e:
        import logging as _logging

        _logging.getLogger(__name__).warning(f"Care plan update email failed: {e}")

    log_audit(db, current_user.id, "care_plan_updated", resource_type="care_plan", resource_id=care_plan_id)
    return _care_plan_to_response(cp, db)


@router.get("/care-plan-templates")
async def get_care_plan_templates(
    current_user: User = Depends(require_clinician()),
):
    """Return predefined care plan templates."""
    return _CARE_PLAN_TEMPLATES
