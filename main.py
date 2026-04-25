"""
DepScreen API — Main Application Entry Point

AI-powered depression screening platform with sentence-level DSM-5
symptom detection, LLM verification, and clinical explanations.

Hardened with: rate limiting, error handling, request logging,
structured logging, LLM retry logic, deep health checks.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

# Silence tokenizer parallelism warnings from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.api.routes import (
    analyze_router,
    auth_router,
    chat_router,
    dashboard_router,
    history_router,
    ingest_router,
    patient_router,
    terminology_router,
    webhooks_router,
)
from app.api.routes.analyze import get_services
from app.core.config import get_settings
from app.core.sentry import init_sentry
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.rate_limiter import limiter
from app.middleware.request_logging import RequestLoggingMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.models.db import init_db

_boot_settings = get_settings()
_log_fmt = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if _boot_settings.log_format != "json"
    else '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
)
logging.basicConfig(
    level=getattr(logging, _boot_settings.log_level.upper(), logging.INFO),
    format=_log_fmt,
)
logger = logging.getLogger(__name__)


# Initialize Sentry BEFORE FastAPI is constructed so any errors during
# startup (DB connection failures, ML model load issues) are captured.
# No-op when SENTRY_DSN is unset.
_settings_for_sentry = get_settings()
init_sentry(
    dsn=_settings_for_sentry.sentry_dsn,
    environment=_settings_for_sentry.environment,
    release=_settings_for_sentry.app_version,
)

# Module-level RAGService instance initialized during lifespan startup.
_rag_service_instance = None  # RAGService instance, set during lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database tables
    init_db()
    logger.info("Database initialized")

    # Pre-initialize ML services
    services = await get_services(settings)
    model_svc = services["model"]

    if model_svc.is_loaded:
        logger.info("Symptom classifier loaded successfully")
    else:
        logger.warning("Symptom classifier not loaded — running in demo mode")

    logger.info(f"LLM model: {settings.llm_model}")
    logger.info(f"Rate limits: auth={settings.rate_limit_auth}, screening={settings.rate_limit_screening}")

    # RAG initialization (embedding model eager-loaded, reranker/NLI lazy)
    # Skip in CI / smoke-test environments to avoid 1GB+ model downloads on startup
    from app.services.container import set_rag_service
    from app.services.rag import RAGService

    global _rag_service_instance
    _rag_service_instance = RAGService(settings)
    if settings.environment not in ("testing", "ci"):
        try:
            await _rag_service_instance.initialize()
            _rag_service_instance.warmup()
            logger.info("RAG service initialized (all models warm)")
        except Exception as e:
            logger.warning(f"RAG service initialization failed (non-fatal): {e}")
    else:
        logger.info("RAG service skipped in testing/CI environment")
    set_rag_service(_rag_service_instance)

    # Start the background scheduler (screening reminders, appointment reminders, care plan reviews)
    try:
        from app.services.scheduler import start_scheduler

        start_scheduler()
    except Exception as e:
        logger.warning(f"Scheduler failed to start (non-fatal): {e}")

    # X/Twitter integration (twikit)
    if settings.x_username and settings.x_email and settings.x_password:
        try:
            from app.services.container import set_x_client
            from app.services.x_client import XClient

            x_client = XClient(
                username=settings.x_username,
                email=settings.x_email,
                password=settings.x_password,
            )
            await x_client.initialize()
            set_x_client(x_client)
            logger.info("X/Twitter client initialized")
        except Exception as e:
            logger.warning(f"X/Twitter client initialization failed (non-fatal): {e}")
    else:
        logger.info("X/Twitter credentials not configured — X analysis disabled")

    yield

    logger.info("Shutting down...")
    try:
        from app.services.scheduler import stop_scheduler

        stop_scheduler()
    except Exception as e:
        logger.warning(f"Scheduler stop failed (non-fatal): {e}")
    await model_svc.unload_models()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="""
        ## DepScreen — AI-Powered Depression Screening API

        Screens free-text input for DSM-5 major depressive episode symptoms
        using a sentence-level transformer classifier, verified by an LLM
        layer for accuracy and safety.

        ### Disclaimer
        This is a screening aid, NOT a diagnostic tool.
        """,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost first) ──

    # 1. Error handler (outermost — catches everything)
    app.add_middleware(ErrorHandlerMiddleware)

    # 2. Security headers — cheap, defensive, attaches to every response
    app.add_middleware(SecurityHeadersMiddleware)

    # 3. Response compression. Gzip payloads >=1KB — covers every list/
    # full-profile response. Roughly 4-6x shrink on typical JSON; saves
    # 150-300ms on mobile/high-latency connections per response.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # 4. Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # 5. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # 6. Rate limiting
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request, exc):
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "You're making requests too quickly. Please wait a moment and try again.",
                    "retry_after": str(exc.detail),
                }
            },
        )

    # ── Routes ──

    app.include_router(auth_router, prefix=f"{settings.api_v1_prefix}/auth", tags=["Authentication"])
    app.include_router(analyze_router, prefix=f"{settings.api_v1_prefix}/analyze", tags=["Screening"])
    app.include_router(history_router, prefix=f"{settings.api_v1_prefix}/history", tags=["History"])
    app.include_router(chat_router, prefix=f"{settings.api_v1_prefix}/chat", tags=["Chat"])
    app.include_router(dashboard_router, prefix=f"{settings.api_v1_prefix}/dashboard", tags=["Dashboard"])
    app.include_router(ingest_router, prefix=f"{settings.api_v1_prefix}/ingest", tags=["Data Ingestion"])
    app.include_router(patient_router, prefix=f"{settings.api_v1_prefix}/patient", tags=["Patient Self-Service"])
    app.include_router(terminology_router, prefix=f"{settings.api_v1_prefix}/terminology", tags=["Terminology"])
    app.include_router(webhooks_router, prefix=f"{settings.api_v1_prefix}/webhooks", tags=["Webhooks"])

    # ── Health Checks ──

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    @app.get("/health/live", tags=["Health"])
    async def liveness():
        """Liveness probe — is the process alive?"""
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness():
        """Readiness probe — are all dependencies operational?"""
        from app.api.routes.analyze import _model_service

        checks = {}

        # ML model
        checks["symptom_model"] = _model_service is not None and _model_service.is_loaded

        # Database
        try:
            from sqlalchemy import text

            from app.models.db import SessionLocal

            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            checks["database"] = True
        except Exception:
            checks["database"] = False

        # RAG — informational; does NOT block readiness
        checks["rag_embedding"] = _rag_service_instance is not None and _rag_service_instance.embedder is not None
        checks["rag_knowledge_base"] = _rag_service_instance is not None and _rag_service_instance.knowledge_base_loaded

        # LLM (check API key is set — don't make a test call)
        checks["llm_configured"] = bool(settings.llm_api_key)

        # RAG checks are informational — exclude from the readiness gate
        readiness_checks = {k: v for k, v in checks.items() if not k.startswith("rag_")}
        all_ready = all(readiness_checks.values())

        return {
            "status": "ready" if all_ready else "degraded",
            "checks": checks,
        }

    # Keep backward-compatible /health endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        from app.api.routes.analyze import _model_service

        return {
            "status": "healthy",
            "symptom_model_loaded": _model_service is not None and _model_service.is_loaded,
        }

    @app.get("/health/llm-test", tags=["Health"])
    async def llm_test():
        """Debug: test LLM call directly.

        Disabled in production — the error path returns `model` and
        `base_url` in the JSON response, which leak provider config
        without adding value to a live deployment. Useful only during
        local dev / staging to spot an auth / DNS / firewall issue fast.
        """
        if settings.is_production:
            raise HTTPException(status_code=404, detail="Not found")

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "Say hello in one word."}],
                max_tokens=10,
            )
            return {
                "status": "ok",
                "response": response.choices[0].message.content or "",
                "model": settings.llm_model,
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
                "model": settings.llm_model,
                "base_url": settings.llm_base_url,
            }

    return app


app = create_app()
