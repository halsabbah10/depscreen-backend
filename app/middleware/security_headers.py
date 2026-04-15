"""
Security headers middleware.

Adds a conservative set of HTTP security headers to every response.

Scope: backend is only reached through the Vercel rewrite proxy, so the
browser never loads a document from the backend origin directly. Most
browser-enforced headers (CSP, X-Frame-Options) ultimately need to be
set by Vercel for the app shell — we set them here too as a second
layer, because:

  - Direct calls to the HF Spaces origin (e.g., during local dev) still
    benefit from these headers.
  - Resend webhook endpoint + PDF downloads serve from the API origin
    and should be framed / sniffed / embedded defensively.
  - Defence-in-depth: if the Vercel config is ever wrong, these headers
    still limit the blast radius.

The CSP is a strict API-appropriate policy — the backend never serves
HTML pages with interactive scripts, so `default-src 'none'` is safe.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_BACKEND_CSP = (
    "default-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach security headers to every outgoing response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        headers = response.headers

        # Prevent MIME-type sniffing — forces browsers to respect the
        # Content-Type we declare (JSON, PDF). Cheap and universally safe.
        headers.setdefault("X-Content-Type-Options", "nosniff")

        # Deny embedding as a frame. API responses should never be iframed.
        headers.setdefault("X-Frame-Options", "DENY")

        # Referrer: don't leak the full API path to third parties on link
        # follow-through. Works even when the browser normally would send
        # the referrer on same-origin.
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

        # Deny browser-feature access the API will never need.
        headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )

        # HSTS — only meaningful on HTTPS, but harmless on HTTP since browsers
        # ignore it there. Max-age 6 months so a misconfiguration is
        # correctable within a reasonable window.
        headers.setdefault(
            "Strict-Transport-Security",
            "max-age=15552000; includeSubDomains",
        )

        # API-tight CSP: backend serves JSON / PDF / file downloads only,
        # never HTML with executable script. Lock it down.
        headers.setdefault("Content-Security-Policy", _BACKEND_CSP)

        return response
