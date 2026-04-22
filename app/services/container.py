"""
Service container — holds singleton service instances initialized at startup.

Route modules import get_rag_service() instead of creating their own RAGService
instances. This prevents the embedding model (~1.3 GB) from being loaded multiple
times when multiple route modules would otherwise each spin up their own instance.

Usage:
  In main.py lifespan:
    from app.services.container import set_rag_service
    set_rag_service(rag_instance)

  In route modules:
    from app.services.container import get_rag_service
    rag_service = get_rag_service()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.rag import RAGService

_rag_service: "RAGService | None" = None


def set_rag_service(service: "RAGService") -> None:
    """Register the singleton RAGService instance (called once at startup)."""
    global _rag_service
    _rag_service = service


def get_rag_service() -> "RAGService | None":
    """Return the singleton RAGService instance, or None if not yet initialized."""
    return _rag_service
