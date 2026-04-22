"""
Authentication and authorization service.

JWT-based auth with bcrypt password hashing and role-based access control.
Roles: patient, clinician, admin.
"""

import logging
import secrets
import string
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.db import AuditLog, User, get_db
from app.services.token_denylist import is_denied

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ── Password Handling ─────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ── Clinician Code Generation ─────────────────────────────────────────────────


def generate_clinician_code(length: int = 6) -> str:
    """Generate a unique alphanumeric invite code for clinicians."""
    chars = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


# ── JWT Token Creation ────────────────────────────────────────────────────────


def create_access_token(
    user_id: str,
    role: str,
    settings: Settings,
) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "jti": str(uuid4()),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    user_id: str,
    settings: Settings,
) -> str:
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": str(uuid4()),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, settings: Settings) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Registration ──────────────────────────────────────────────────────────────


def register_user(
    email: str,
    password: str,
    full_name: str,
    role: str,
    db: Session,
    clinician_code: str | None = None,
) -> User:
    """Register a new user.

    - Clinicians get an auto-generated invite code.
    - Patients can optionally link to a clinician via their code.
    """
    # Validate role
    if role not in ("patient", "clinician"):
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'clinician'")

    # Check email uniqueness
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Validate password strength
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Resolve clinician link
    clinician_id = None
    if clinician_code and role == "patient":
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
        clinician_id = clinician.id

    # Generate clinician code if registering as clinician
    code = None
    if role == "clinician":
        code = generate_clinician_code()
        # Ensure uniqueness
        while db.query(User).filter(User.clinician_code == code).first():
            code = generate_clinician_code()

    user = User(
        id=str(uuid4()),
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role=role,
        clinician_id=clinician_id,
        clinician_code=code,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"Registered user {user.id} ({role})")
    return user


# ── Login ─────────────────────────────────────────────────────────────────────


def authenticate_user(email: str, password: str, db: Session) -> User:
    """Authenticate user by email and password."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return user


# ── Dependencies (FastAPI Depends) ────────────────────────────────────────────


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate the current user from the JWT token."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token, settings)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Check if token has been revoked
    jti = payload.get("jti")
    if jti and await is_denied(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    return user


def require_role(*allowed_roles: str):
    """Dependency factory that checks the user's role.

    Usage: Depends(require_role("clinician", "admin"))
    """

    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(allowed_roles)}",
            )
        return current_user

    return role_checker


def require_patient():
    """Shorthand for require_role('patient')."""
    return require_role("patient")


def require_clinician():
    """Shorthand for require_role('clinician', 'admin')."""
    return require_role("clinician", "admin")


# ── Audit Logging ─────────────────────────────────────────────────────────────


def log_audit(
    db: Session,
    user_id: str,
    action: str,
    resource_type: str = "",
    resource_id: str | None = None,
    ip_address: str | None = None,
):
    """Write an entry to the audit log."""
    entry = AuditLog(
        id=str(uuid4()),
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
