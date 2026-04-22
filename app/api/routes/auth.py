"""
Authentication API routes.

Registration, login, token refresh, and profile management.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import User, get_db
from app.schemas.analysis import (
    LoginRequest,
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
from app.services.token_denylist import deny_token

router = APIRouter()
logger = logging.getLogger(__name__)


def _set_refresh_cookie(response: JSONResponse, token: str, settings: Settings) -> None:
    """Set the refresh token as an httpOnly cookie on the response."""
    response.set_cookie(
        key="refresh_token",
        value=token,
        httponly=True,
        secure=settings.is_production,
        samesite="strict" if settings.is_production else "lax",
        path="/api/auth",
        max_age=settings.refresh_token_expire_days * 86400,
    )


def _clear_refresh_cookie(response: JSONResponse) -> None:
    """Delete the refresh token cookie."""
    response.delete_cookie(key="refresh_token", path="/api/auth")


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


@router.post("/register")
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

    body_out = TokenResponse(
        access_token=access_token,
        user=_user_to_profile(user),
    )
    response = JSONResponse(content=body_out.model_dump(mode="json"))
    _set_refresh_cookie(response, refresh_token, settings)
    return response


@router.post("/login")
@limiter.limit("10/minute")
async def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Authenticate and receive JWT tokens."""
    from datetime import UTC
    from datetime import datetime as _dt

    user = authenticate_user(body.email, body.password, db)

    # Track last_login_at for audit / clinician view
    user.last_login_at = _dt.now(UTC)
    db.commit()

    access_token = create_access_token(user.id, user.role, settings)
    refresh_token = create_refresh_token(user.id, settings)

    log_audit(db, user.id, "login", resource_type="user")

    body_out = TokenResponse(
        access_token=access_token,
        user=_user_to_profile(user),
    )
    response = JSONResponse(content=body_out.model_dump(mode="json"))
    _set_refresh_cookie(response, refresh_token, settings)
    return response


@router.post("/refresh")
async def refresh_token(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Refresh an access token using the httpOnly refresh cookie."""
    cookie_token = request.cookies.get("refresh_token")
    if not cookie_token:
        raise HTTPException(status_code=401, detail="No refresh token cookie")
    payload = decode_token(cookie_token, settings)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type — expected refresh token")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    new_access = create_access_token(user.id, user.role, settings)
    new_refresh = create_refresh_token(user.id, settings)

    body_out = TokenResponse(
        access_token=new_access,
        user=_user_to_profile(user),
    )
    response = JSONResponse(content=body_out.model_dump(mode="json"))
    _set_refresh_cookie(response, new_refresh, settings)
    return response


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_user)):
    """Get the currently authenticated user's profile."""
    return _user_to_profile(current_user)


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Server-side logout — revokes tokens and clears refresh cookie."""
    # Deny the access token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        access_token_str = auth_header.removeprefix("Bearer ")
        try:
            access_payload = decode_token(access_token_str, settings)
            if access_jti := access_payload.get("jti"):
                from datetime import UTC, datetime

                exp = datetime.fromtimestamp(access_payload["exp"], tz=UTC)
                await deny_token(access_jti, exp)
        except Exception:
            logger.debug("Access token deny skipped (already expired or invalid)")

    # Deny the refresh token (from cookie)
    refresh_cookie = request.cookies.get("refresh_token")
    if refresh_cookie:
        try:
            refresh_payload = decode_token(refresh_cookie, settings)
            if refresh_jti := refresh_payload.get("jti"):
                from datetime import UTC, datetime

                exp = datetime.fromtimestamp(refresh_payload["exp"], tz=UTC)
                await deny_token(refresh_jti, exp)
        except Exception:
            logger.debug("Refresh token deny skipped (cookie expired or invalid)")

    log_audit(db, current_user.id, "logout", resource_type="user")
    response = JSONResponse(content={"status": "logged_out"})
    _clear_refresh_cookie(response)
    return response


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
