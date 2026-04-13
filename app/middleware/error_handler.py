"""
Global error handler.

Catches unhandled exceptions and returns sanitized JSON responses.
Never leaks stack traces, file paths, or internal state to the client.
Logs the full traceback server-side for debugging.

For a mental health platform: error messages must be gentle and reassuring.
"""

import logging
import traceback
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4())[:8])

        try:
            response = await call_next(request)
            # Attach request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            # Log the full traceback server-side
            logger.error(
                f"Unhandled exception [request_id={request_id}] "
                f"{request.method} {request.url.path}: {exc}",
                exc_info=True,
            )

            # Return sanitized response — NEVER expose internals
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Something went wrong on our end. Your data is safe. Please try again in a moment.",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )
