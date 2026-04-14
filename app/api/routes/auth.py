"""
Authentication API routes.

Registration, login, token refresh, and profile management.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import User, get_db
from app.schemas.analysis import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserProfile,
)
from app.services.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    log_audit,
    register_user,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _user_to_profile(user: User) -> UserProfile:
    """Convert a User DB model to a public UserProfile schema."""
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        clinician_code=user.clinician_code,
        created_at=user.created_at,
        onboarding_completed=user.onboarding_completed or False,
        profile_picture_url=user.profile_picture_url,
        language_preference=user.language_preference,
        reddit_username=user.reddit_username,
        twitter_username=user.twitter_username,
        date_of_birth=user.date_of_birth.isoformat() if user.date_of_birth else None,
        gender=user.gender,
        nationality=user.nationality,
        cpr_number=user.cpr_number,
        phone=user.phone,
        blood_type=user.blood_type,
        timezone=user.timezone,
    )


@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register(
    body: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Register a new patient or clinician account."""
    user = register_user(
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        role=body.role,
        db=db,
        clinician_code=body.clinician_code,
    )

    access_token = create_access_token(user.id, user.role, settings)
    refresh_token = create_refresh_token(user.id, settings)

    log_audit(db, user.id, "register", resource_type="user")

    # Fire-and-forget welcome email (silent if email service not configured)
    if user.role == "patient":
        try:
            from app.services.email import get_email_service

            get_email_service(settings).send_welcome(user.full_name, user.email)
        except Exception as e:
            logger.warning(f"Welcome email failed: {e}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=_user_to_profile(user),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Authenticate and receive JWT tokens."""
    from datetime import datetime as _dt

    user = authenticate_user(body.email, body.password, db)

    # Track last_login_at for audit / clinician view
    user.last_login_at = _dt.utcnow()
    db.commit()

    access_token = create_access_token(user.id, user.role, settings)
    refresh_token = create_refresh_token(user.id, settings)

    log_audit(db, user.id, "login", resource_type="user")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=_user_to_profile(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Refresh an access token using a valid refresh token."""
    payload = decode_token(body.refresh_token, settings)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type — expected refresh token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    new_access = create_access_token(user.id, user.role, settings)
    new_refresh = create_refresh_token(user.id, settings)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        user=_user_to_profile(user),
    )


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return _user_to_profile(current_user)


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Server-side logout — records the audit event.

    Note: stateless JWTs can't be truly revoked without a denylist. For now,
    we log the event and rely on the client clearing tokens. A production
    deployment should add a Redis-backed denylist keyed by jti claim.
    """
    log_audit(db, current_user.id, "logout", resource_type="user")
    return {"status": "logged_out"}


class LinkClinicianRequest(BaseModel):
    clinician_code: str = Field(min_length=4, max_length=12)


@router.post("/link")
async def link_to_clinician(
    body: LinkClinicianRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Link a patient to a clinician using their invite code."""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can link to a clinician")
    clinician_code = body.clinician_code

    clinician = (
        db.query(User)
        .filter(
            User.clinician_code == clinician_code,
            User.role == "clinician",
        )
        .first()
    )
    if not clinician:
        raise HTTPException(status_code=404, detail="Invalid clinician code")

    current_user.clinician_id = clinician.id
    db.commit()

    log_audit(db, current_user.id, "linked_to_clinician", resource_type="user", resource_id=clinician.id)

    return {
        "status": "linked",
        "clinician_name": clinician.full_name,
    }
