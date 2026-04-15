"""Application configuration using pydantic-settings."""

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── API ────────────────────────────────────────────────────────────────
    app_name: str = "DepScreen"
    app_version: str = "2.0.0"
    debug: bool = False
    api_v1_prefix: str = "/api"
    environment: str = "development"  # development, staging, production

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── LLM ────────────────────────────────────────────────────────────────
    llm_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3

    # ── Email (Resend) ─────────────────────────────────────────────────────
    resend_api_key: str = ""
    resend_webhook_secret: str = ""  # svix-compatible; paste from Resend dashboard after adding the webhook endpoint
    email_from: str = "DepScreen <onboarding@resend.dev>"
    email_enabled: bool = True  # master switch — emails silently skip if resend_api_key is empty

    # ── Supabase Storage (profile pictures) ────────────────────────────────
    # Uses the service role key; never expose to the frontend.
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_avatar_bucket: str = "depscreen-avatars"

    # ── Error monitoring (Sentry) ──────────────────────────────────────────
    # If unset, Sentry initializes as a no-op (local dev, CI).
    sentry_dsn: str = ""

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./app.db"

    # ── Model Paths ────────────────────────────────────────────────────────
    model_path: Path = Path("./ml/models")
    symptom_model_name: str = "symptom_classifier.pt"
    symptom_metadata_name: str = "redsm5_metadata.json"

    # ── Auth ───────────────────────────────────────────────────────────────
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60  # 1 hour (reduced from 24h)
    refresh_token_expire_days: int = 7

    @field_validator("jwt_secret", mode="before")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        if not v or v in ("", "depscreen-dev-secret-change-in-production", "depscreen-docker-secret"):
            # Auto-generate a secure secret for development
            # In production, this MUST be set via environment variable
            return secrets.token_hex(32)
        return v

    # ── RAG ─────────────────────────────────────────────────────────────────
    knowledge_base_dir: Path = Path("./ml/knowledge_base")

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: str = '["http://localhost:3000","http://localhost:5173"]'

    # ── Rate Limiting ──────────────────────────────────────────────────────
    rate_limit_auth: str = "10/minute"  # Login/register attempts
    rate_limit_screening: str = "20/minute"  # Screening submissions
    rate_limit_chat: str = "30/minute"  # Chat messages
    rate_limit_default: str = "100/minute"  # General endpoints

    # ── Safety ─────────────────────────────────────────────────────────────
    always_include_disclaimer: bool = True
    max_text_length: int = 10000

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"  # json or text

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from the configured JSON string.

        In production we never fall back to the localhost defaults — a
        malformed `CORS_ORIGINS` should fail loudly rather than silently
        expose the API to localhost from anywhere. In dev the fallback
        keeps `npm run dev` on :3000 / :5173 working without extra env.
        """
        import json

        try:
            origins = json.loads(self.cors_origins)
            if not isinstance(origins, list):
                raise ValueError("CORS_ORIGINS must be a JSON array")
            return origins
        except (json.JSONDecodeError, ValueError) as e:
            if self.is_production:
                raise RuntimeError(f"CORS_ORIGINS must be a valid JSON array in production: {e}") from e
            return ["http://localhost:3000", "http://localhost:5173"]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
