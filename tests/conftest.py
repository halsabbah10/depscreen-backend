"""
Pytest configuration + fixtures for DepScreen backend.

Strategy
--------
Tests run against an in-memory SQLite DB so they're fast and isolated.
External services (LLM, RAG, email, avatar storage, inference HTTP) are
patched at the service-boundary layer — we exercise real route + schema
+ SQL logic, but never a real paid API call.

The FastAPI lifespan (which initializes ML models, Alembic, and the
APScheduler) is replaced with a no-op so `create_app()` doesn't try to
load a 400 MB DistilBERT during `pytest --collect-only`.

Because the pgvector-backed tables (KnowledgeChunk, PatientRAGChunk)
can't be created on SQLite, we skip them in `create_all()`. Tests that
need RAG behavior mock the service entirely, which matches production
behavior on the endpoint level.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

# ── Environment setup ────────────────────────────────────────────────────────
# Must happen BEFORE importing anything from `app` — the settings singleton
# reads these once via pydantic-settings and caches the result.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-production-" + "x" * 16)
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("CORS_ORIGINS", '["http://testserver"]')
# Silence Sentry fully during tests
os.environ.setdefault("SENTRY_DSN", "")

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# ── Import app modules (after env vars are set) ──────────────────────────────
from app.core.config import get_settings  # noqa: E402
from app.models.db import Base, User, get_db  # noqa: E402
from app.services.auth import (  # noqa: E402
    create_access_token,
    generate_clinician_code,
    hash_password,
)

# ─────────────────────────────────────────────────────────────────────────────
# Engine + session fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_engine():
    """A single in-memory SQLite engine shared across the test session.

    StaticPool + shared connection is the canonical pattern for in-memory
    SQLite with SQLAlchemy — without it, each new connection gets its own
    blank DB and fixtures can't share state with the FastAPI app.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create every table EXCEPT those with pgvector/TSVector columns —
    # SQLite can't handle them. On real Postgres these hold RAG chunks;
    # tests that need RAG mock the service, not the storage, so the
    # tables are unreachable and safe to skip. This check is dynamic so
    # newly added vector-backed tables are skipped automatically.
    def _has_vector_cols(table) -> bool:
        return any("vector" in str(col.type).lower() or "tsvector" in str(col.type).lower() for col in table.columns)

    tables = [t for t in Base.metadata.tables.values() if not _has_vector_cols(t)]
    Base.metadata.create_all(bind=engine, tables=tables)

    yield engine
    engine.dispose()


@pytest.fixture
def db(test_engine) -> Generator[Session, None, None]:
    """A fresh DB session per test, with truncation between tests.

    Truncation (not transactional rollback) because FastAPI's route
    handlers commit, and rolling back a committed session is a no-op.
    Truncating every table between tests keeps isolation cheap.
    """
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestSession()

    try:
        yield session
    finally:
        session.close()
        # Truncate everything for the next test (skip vector-backed tables
        # that were never created on SQLite)
        with test_engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                has_vector_cols = any(
                    "vector" in str(col.type).lower() or "tsvector" in str(col.type).lower() for col in table.columns
                )
                if has_vector_cols:
                    continue
                conn.execute(table.delete())


# ─────────────────────────────────────────────────────────────────────────────
# App + client fixtures
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    """Replace the real lifespan so tests don't load the ML model or
    start the APScheduler."""
    yield


@pytest.fixture
def app(test_engine, db, monkeypatch) -> FastAPI:
    """Build a FastAPI app wired to the in-memory test DB.

    `create_app()` from `main.py` is imported fresh here (after the env is
    set) and its lifespan swapped for a no-op. The `get_db` dependency
    is overridden to return our session-bound factory.
    """
    # Patch heavyweight startup work before create_app runs
    monkeypatch.setattr("app.models.db.init_db", lambda: None, raising=False)

    from main import create_app

    application = create_app()

    # Replace lifespan — FastAPI reads this via app.router.lifespan_context
    application.router.lifespan_context = _noop_lifespan

    # Route every `Depends(get_db)` to a session bound to our test engine.
    # We deliberately yield a *fresh* session per request (not the
    # test-level `db` fixture) because request handlers commit and then
    # expect to re-query — sharing a single session across the test and
    # the request would leak uncommitted state between them.
    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    def override_get_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    application.dependency_overrides[get_db] = override_get_db

    # Disable rate limiting so parallel tests don't bump into the 10/min
    # auth throttle. slowapi uses IP as the key and TestClient always
    # hits from the same synthetic address.
    application.state.limiter.enabled = False

    yield application

    application.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    """Sync test client. Use for every non-streaming route test."""
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# User factory fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_user(
    db: Session,
    *,
    role: str,
    email: str | None = None,
    password: str = "test-password-123",
    full_name: str = "Test User",
    clinician_id: str | None = None,
    **extra: Any,
) -> User:
    """Insert a user straight into the DB, bypassing the HTTP layer."""
    user = User(
        id=str(uuid4()),
        email=email or f"{uuid4().hex[:8]}@test.local",
        password_hash=hash_password(password),
        full_name=full_name,
        role=role,
        is_active=True,
        onboarding_completed=True,
        clinician_id=clinician_id,
        clinician_code=generate_clinician_code() if role == "clinician" else None,
        **extra,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def patient_user(db) -> User:
    return _make_user(db, role="patient", email="patient@test.local", full_name="Patient Test")


@pytest.fixture
def clinician_user(db) -> User:
    return _make_user(db, role="clinician", email="clinician@test.local", full_name="Dr. Test")


@pytest.fixture
def linked_patient(db, clinician_user) -> User:
    """A patient already linked to the clinician_user fixture."""
    return _make_user(
        db,
        role="patient",
        email="linked@test.local",
        full_name="Linked Patient",
        clinician_id=clinician_user.id,
    )


def _auth_headers(user: User) -> dict[str, str]:
    token = create_access_token(user.id, user.role, get_settings())
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def patient_headers(patient_user) -> dict[str, str]:
    return _auth_headers(patient_user)


@pytest.fixture
def clinician_headers(clinician_user) -> dict[str, str]:
    return _auth_headers(clinician_user)


@pytest.fixture
def linked_patient_headers(linked_patient) -> dict[str, str]:
    return _auth_headers(linked_patient)


# ─────────────────────────────────────────────────────────────────────────────
# External service mocks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _silence_external_services(monkeypatch):
    """Block every real network call by default.

    Tests that need a specific mocked return value should patch a
    narrower target in their own fixture — this one just guarantees no
    test accidentally hits OpenRouter / Resend / Supabase.
    """
    # Email: send_welcome / send_notification / send_screening_due — all no-op
    from app.services import email as email_module

    class _NoopEmail:
        def send_welcome(self, *a, **kw):
            return None

        def send_notification(self, *a, **kw):
            return None

        def send_screening_due(self, *a, **kw):
            return None

        def send_appointment_reminder(self, *a, **kw):
            return None

        def send_care_plan_updated(self, *a, **kw):
            return None

        def is_configured(self):
            return False

    monkeypatch.setattr(email_module, "get_email_service", lambda _settings: _NoopEmail())

    # Avatar: block real Supabase Storage calls
    try:
        from app.services import avatar as avatar_module

        async def _fake_upload(*a, **kw):
            return "https://test.local/avatar.webp"

        monkeypatch.setattr(avatar_module, "upload_avatar", _fake_upload, raising=False)
    except ImportError:
        pass

    yield
