"""
Profile picture storage — Supabase Storage backend.

Handles the round-trip: accept a raw image upload, normalise it (resize to
512x512 square, strip EXIF for privacy, re-encode as WebP), and push to
Supabase Storage under a stable per-user path. Returns the public URL we
stash on users.profile_picture_url.

Uses the service role key — only the backend touches this module, the
frontend never receives it.

Design choices:
- Fixed 512x512 square: everywhere we render avatars we render them
  circular at <= 56px, so the upload target is aggressive-but-not-overkill.
- Centre-crop to square rather than letterbox: avatars are always shown
  as circles; empty space at the top/bottom would bulge awkwardly.
- WebP output: ~30% smaller than JPEG at equivalent quality, and Supabase
  serves it with correct Content-Type automatically.
- EXIF stripped: drops GPS, device serial, the original filename, etc.
- Deterministic key: `{user_id}/avatar.webp`. One active avatar per user,
  new uploads overwrite the old — no orphan files accumulating.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image, ImageOps
from supabase import Client, create_client

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ── Image normalisation ─────────────────────────────────────────────────────

TARGET_SIZE = 512
WEBP_QUALITY = 82
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB upload cap
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


class AvatarError(ValueError):
    """Raised for input problems we want the client to see."""


def _normalize_to_webp(data: bytes) -> bytes:
    """Open, centre-crop, resize, and re-encode a user upload as WebP.

    Raises AvatarError with a friendly message on any decoding problem.
    Never raises for valid-but-weird images — rotates according to EXIF
    orientation before stripping, so sideways iPhone photos come out
    upright.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise AvatarError(
            f"That image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please upload a smaller file."
        )

    try:
        img = Image.open(io.BytesIO(data))
        # Apply EXIF rotation BEFORE stripping metadata, so phones-held-sideways
        # are corrected. `exif_transpose` handles all eight orientations.
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise AvatarError("We couldn't read that image. Try a JPG, PNG, or WebP.") from e

    # Coerce to RGB — WebP supports alpha but the avatar frame is circular
    # so transparent backgrounds never show through. Dropping alpha keeps
    # files smaller and avoids odd fringing around dark avatars on cream cards.
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Centre-crop to square
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))

    # Resize to target. LANCZOS is the sharpest-available downscaler in Pillow.
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    # `method=6` is slower encode but smaller output — worth it since
    # uploads are infrequent and the file is downloaded every page.
    img.save(out, format="WEBP", quality=WEBP_QUALITY, method=6)
    return out.getvalue()


# ── Supabase client ──────────────────────────────────────────────────────────

_client: Client | None = None
_bucket_checked = False


def _get_client(settings: Settings) -> Client | None:
    """Lazy singleton. Returns None if Supabase isn't configured — callers
    should treat that as 'avatar feature disabled'."""
    global _client
    if _client is not None:
        return _client
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _client


def _ensure_bucket(client: Client, bucket: str) -> None:
    """Create the avatar bucket as public on first use. Idempotent after
    the first request per process."""
    global _bucket_checked
    if _bucket_checked:
        return
    try:
        existing = {b.name for b in client.storage.list_buckets()}
        if bucket not in existing:
            logger.info(f"Creating Supabase Storage bucket: {bucket}")
            client.storage.create_bucket(
                bucket,
                options={"public": True, "file_size_limit": str(MAX_UPLOAD_BYTES)},
            )
    except Exception as e:
        # Don't crash on bucket probing; the upload call below will surface
        # a clearer error if the bucket genuinely can't be used.
        logger.warning(f"Bucket probe for {bucket} failed (will try upload anyway): {e}")
    _bucket_checked = True


# ── Public API ──────────────────────────────────────────────────────────────


class AvatarService:
    """Wrapper around Supabase Storage for per-user avatar management."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bucket = settings.supabase_avatar_bucket
        self._client = _get_client(settings)
        self.enabled = self._client is not None
        if not self.enabled:
            logger.info("Avatar service disabled — SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    def upload(self, user_id: str, raw_bytes: bytes, content_type: str | None = None) -> str:
        """Normalise + upload the avatar for a user. Returns the public URL.

        Overwrites any existing avatar at the deterministic key. Raises
        AvatarError for input problems (too big, can't decode). Raises
        RuntimeError if Supabase is not configured.
        """
        if not self._client:
            raise RuntimeError("Avatar storage is not configured on this server.")

        if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
            raise AvatarError("Only JPG, PNG, WebP, and HEIC images are supported.")

        webp_bytes = _normalize_to_webp(raw_bytes)

        _ensure_bucket(self._client, self.bucket)

        key = f"{user_id}/avatar.webp"
        try:
            # `upsert=True` so repeated uploads overwrite without 409.
            self._client.storage.from_(self.bucket).upload(
                path=key,
                file=webp_bytes,
                file_options={
                    "content-type": "image/webp",
                    "upsert": "true",
                    # 1-year cache — filename is stable per user so we add a
                    # cache-buster (the updated_at timestamp) at the URL level
                    # to force clients to pick up new uploads.
                    "cache-control": "public, max-age=31536000, immutable",
                },
            )
        except Exception as e:
            logger.error(f"Supabase upload failed for user {user_id}: {e}")
            raise RuntimeError("We couldn't save your picture. Please try again in a moment.") from e

        # Public URL from the bucket. Supabase returns {"publicUrl": "..."}.
        url_resp: Any = self._client.storage.from_(self.bucket).get_public_url(key)
        public_url = url_resp if isinstance(url_resp, str) else url_resp.get("publicUrl", "")
        if not public_url:
            raise RuntimeError("Upload succeeded but the public URL is missing.")

        # URL is stable across uploads, so clients would cache the old image.
        # Append a version query-string driven by upload time so the browser
        # reliably pulls the new file.
        import time

        return f"{public_url}?v={int(time.time())}"

    def delete(self, user_id: str) -> None:
        """Remove the user's avatar. No-ops if it isn't there."""
        if not self._client:
            return
        key = f"{user_id}/avatar.webp"
        try:
            self._client.storage.from_(self.bucket).remove([key])
        except Exception as e:
            logger.warning(f"Supabase avatar delete for {user_id} failed (ignoring): {e}")


# ── Singleton accessor ──────────────────────────────────────────────────────

_service: AvatarService | None = None


def get_avatar_service(settings: Settings) -> AvatarService:
    global _service
    if _service is None:
        _service = AvatarService(settings)
    return _service
