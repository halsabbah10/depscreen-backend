"""
Request logging middleware.

Logs every request with: method, path, status code, duration, user ID (if authenticated).
Uses structured JSON format for production observability.
"""

import logging
import time
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("depscreen.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4())[:8])
        start_time = time.time()

        # Extract user ID from JWT if present (without full auth — just peek)
        user_id = "anonymous"
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and len(auth_header) > 20:
            user_id = "authenticated"  # Don't decode JWT here — just note it exists

        response = await call_next(request)

        duration_ms = round((time.time() - start_time) * 1000, 1)
        status = response.status_code

        # Skip health checks and static assets from logs
        path = request.url.path
        if path in ("/", "/health", "/health/live", "/health/ready", "/favicon.ico"):
            return response

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms,
            "user": user_id,
            "ip": request.client.host if request.client else "unknown",
        }

        if status >= 500:
            logger.error(f"[{request_id}] {request.method} {path} → {status} ({duration_ms}ms)", extra=log_data)
        elif status >= 400:
            logger.warning(f"[{request_id}] {request.method} {path} → {status} ({duration_ms}ms)", extra=log_data)
        else:
            logger.info(f"[{request_id}] {request.method} {path} → {status} ({duration_ms}ms)", extra=log_data)

        return response
