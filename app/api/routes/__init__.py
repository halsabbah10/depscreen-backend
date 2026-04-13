"""API route modules."""

from .analyze import router as analyze_router
from .auth import router as auth_router
from .chat import router as chat_router
from .dashboard import router as dashboard_router
from .history import router as history_router
from .ingest import router as ingest_router
from .patient import router as patient_router

__all__ = [
    "auth_router",
    "analyze_router",
    "history_router",
    "chat_router",
    "dashboard_router",
    "ingest_router",
    "patient_router",
]
