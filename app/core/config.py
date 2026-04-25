"""Application configuration using pydantic-settings."""

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
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
    llm_model: str = "gemini-3-flash-preview"  # Default / fallback
    llm_model_pro: str = "gemini-3.1-pro-preview"  # Reasoning: adversarial, explanation, secondary symptoms
    llm_model_flash: str = "gemini-3-flash-preview"  # Balanced: chat, evidence validation, confidence
    llm_model_lite: str = "gemini-3.1-flash-lite-preview"  # Utility: auto-title, lightweight tasks
    llm_timeout_seconds: int = 120  # Bumped for Pro reasoning latency
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

    # ── X/Twitter integration (twikit) ────────────────────────────────
    # All three must be set to enable X analysis. If any are empty,
    # the /ingest/x endpoint returns 503.
    x_username: str = ""
    x_email: str = ""
    x_password: str = ""

    # ── Error monitoring (Sentry) ──────────────────────────────────────────
    # If unset, Sentry initializes as a no-op (local dev, CI).
    sentry_dsn: str = ""

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "postgresql://postgres:postgres@localhost:5432/depscreen"

    # ── Model Paths ────────────────────────────────────────────────────────
    model_path: Path = Path("./ml/models/v_production_ensemble")
    symptom_model_name: str = "symptom_classifier.pt"
    symptom_metadata_name: str = "redsm5_metadata.json"
    hf_model_repo: str = "halsabbah/depscreen-models"  # HF model repo for ensemble weights

    # ── Auth ───────────────────────────────────────────────────────────────
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60  # 1 hour (reduced from 24h)
    refresh_token_expire_days: int = 7

    @model_validator(mode="after")
    def validate_jwt_secret(self) -> "Settings":
        known_defaults = ("", "depscreen-dev-secret-change-in-production", "depscreen-docker-secret")
        if not self.jwt_secret or self.jwt_secret in known_defaults:
            if self.environment == "production":
                raise ValueError(
                    "JWT_SECRET must be explicitly set in production. "
                    'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
                )
            self.jwt_secret = secrets.token_hex(32)
        return self

    @model_validator(mode="after")
    def validate_database_url(self) -> "Settings":
        if self.environment == "production" and ("localhost" in self.database_url or "sqlite" in self.database_url):
            raise ValueError(
                "DATABASE_URL must not use localhost or sqlite in production. Set a real PostgreSQL connection string."
            )
        return self

    # ── RAG ─────────────────────────────────────────────────────────────────
    knowledge_base_dir: Path = Path("./ml/knowledge_base")

    # Models (provisional — validated before deployment)
    rag_embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5"
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_nli_model: str = "cross-encoder/nli-deberta-v3-base"
    rag_embedding_dimensions: int = 1024

    # Retrieval
    rag_similarity_threshold: float = 0.40  # Placeholder — calibrate empirically
    rag_retrieval_top_k: int = 20  # Bi-encoder + BM25 candidates each
    rag_rerank_top_k: int = 7  # Post-reranker results
    rag_rrf_k: int = 50
    rag_full_text_weight: float = 0.7
    rag_semantic_weight: float = 1.0

    # Token budget (percentages apply AFTER structured data deducted)
    rag_max_context_tokens: int = 2500
    rag_budget_clinical: float = 0.45
    rag_budget_patient_history: float = 0.40
    rag_budget_buffer: float = 0.15

    # Chunking
    rag_child_chunk_target: int = 400
    rag_child_chunk_max: int = 512
    rag_child_chunk_overlap: int = 64
    rag_parent_chunk_max: int = 1500

    # Chat summary
    rag_chat_extraction_threshold: float = 0.45
    rag_chat_summary_min_messages: int = 10
    rag_chat_summary_min_substantive: int = 3

    # Timeouts
    rag_timeout_screening: float = 5.0
    rag_timeout_chat: float = 2.0
    rag_timeout_crisis: float = 0.0

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: str = '["http://localhost:3000","http://localhost:5173"]'

    # ── Rate Limiting ──────────────────────────────────────────────────────
    rate_limit_auth: str = "10/minute"  # Login/register attempts
    rate_limit_screening: str = "20/minute"  # Screening submissions
    rate_limit_chat: str = "30/minute"  # Chat messages
    rate_limit_default: str = "100/minute"  # General endpoints
    redis_url: str = ""  # Upstash Redis URL; empty = in-memory fallback

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
