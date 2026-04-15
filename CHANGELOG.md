# Changelog

All notable changes to the DepScreen backend. Dates in DD/MM/YYYY (Asia/Bahrain).

## [Unreleased] — 2026-04-16

### Security
- `/health/llm-test` now returns 404 in production — previously leaked `LLM_API_KEY`, `base_url`, and model name in error responses. Live locally for quick LLM-config smoke tests only.
- `get_cors_origins()` fails loudly on malformed `CORS_ORIGINS` in production instead of silently falling back to `localhost:3000,5173`. Dev behavior unchanged.
- Reddit / X ingestion error messages referenced the wrong variable (`request.username` — a FastAPI Request object, not the body). Corrected to `body.username`, so the 404 message now actually shows the username the user submitted.

### Tests
- Pytest suite added: **140 tests, 36% coverage, 35% gate** in CI.
  - Route-level: auth (16), screening (9), chat CRUD (10), patient CRUD (14), dashboard (11)
  - Service-level: DecisionService (19, every confidence + flagging branch), SafetyGuard (24, every redaction category), Bahrain localization + crisis keyword lexicon (37)
- In-memory SQLite, services mocked at the `Depends` boundary; rate limiter disabled; FastAPI lifespan replaced with no-op so tests don't load the 400 MB DistilBERT.
- Wired into `.github/workflows/ci.yml` — pytest must pass before Docker build.

### Docs
- `README.md` full rewrite — covers the 6-step screening pipeline, 4 safety layers, required + optional env vars, endpoints grouped by audience, Alembic workflow, HF Spaces deployment. HF Space YAML frontmatter moved to `.hf-space-header.md` and injected only on deploy (kept out of the GitHub view).
- `TESTING.md` added.

## [2026-04-15] — Performance pass

### Added
- **gzip compression** on all responses ≥1 KB (`GZipMiddleware`). Shrinks typical JSON 4–6× on the wire; saves 150–300 ms on mobile / high-latency connections.
- **Hot-path B-tree indexes** on the 5 columns most-filtered on every request: `medications(is_active)`, `care_plans(status)`, `notifications(user_id, is_read)`, `allergies(severity)`, `screenings(patient_id, created_at DESC)`.
- **pgvector HNSW indexes** on `knowledge_chunks.embedding` and `patient_rag_chunks.embedding` — O(log n) vector search instead of full-table scan.
- **Connection pool bump** to 20 + 40 overflow (was 10 + 20) and `pool_recycle=1800` to stay ahead of Supabase session-pooler idle timeouts. `pool_pre_ping` kept.
- **Alembic skip-if-current** in `init_db()` so idle pods don't re-run migrations on every boot.

### Changed
- `/dashboard/stats`: **5 queries → 2**. One aggregation row with `func.sum(case(...))` for flagged / this-week / severity buckets instead of four separate counts plus a GROUP BY.
- `/dashboard/patients`: **(1 + 2N) queries → 3**. Latest-screening-per-patient via `max(created_at)` subquery JOIN; counts via single GROUP BY. Scales flat regardless of patient count.

### Fixed
- Crisis-alert email deep-link pointed clinicians to the patient-only `/screening/{id}` route; now links to the shared `/results/{id}`.

### Deps
- CI pins bumped to Node-24-compatible actions.
- pydantic unpinned so `svix>=1.20` resolves its `ModelWrapValidatorHandler` import cleanly.

## [2026-04-14] — Phase H: Enterprise hardening

- **Sentry** error monitoring wired (FastAPI integration; no-op when DSN unset).
- **Security headers** middleware (CSP, HSTS, X-Content-Type-Options, Referrer-Policy, Permissions-Policy).
- **Supabase Storage** avatar upload — 5 MB limit, Pillow resize to 512×512, WebP conversion, EXIF stripped.
- **Alembic** formalized — idempotent boot-time migration with fresh-DB stamp path.
- **Dependency audit** in CI (`pip-audit`) with scoped CVE ignore list.
- **Resend webhook** delivery tracking — queued → sent → delivered → bounced state machine.

## [2026-04-14] — Phase D: PDF exports

- Screening result PDF (`/api/analyze/{id}/export.pdf`).
- Patient data PDF (`/api/patient/export.pdf`).
- Clinical summary PDF for clinicians (`/api/dashboard/patients/{id}/summary.pdf`).
- All powered by `reportlab`, matching the Clinical Sanctuary design language (Cormorant Garamond headings, warm cream backgrounds).

## [2026-04-14] — Phase C: Smart form inputs

- RxNorm drug name autocomplete (`/api/terminology/drugs`) — free NLM endpoint, no API key.
- ICD-10 code lookup (`/api/terminology/icd10`) — same provider.
- PDF form extraction via `pdfplumber` — uploaded PHQ-9 / GAD-7 forms parsed for scores.

## [2026-04-14] — Phase B: Clinician workflow + safety

- **SafetyGuard** — regex-based output redaction for LLM responses. Four categories: prescription advice, diagnostic claims, self-harm encouragement, anti-professional content. Replaces matched phrases with warm substitutes + appends Bahrain crisis disclaimer.
- **Crisis response** tone rewrite — LLM-driven warm response instead of a cold canned resource dump. Falls back to the static Bahrain response on LLM failure.
- **Clinician dashboard**: patient full-profile endpoint, screening triage, flagged-cases view.
- **APScheduler** background jobs: screening reminders, appointment reminders, care-plan reviews.
- **Patient-facing** appointment + care-plan read endpoints.
- **Chat**: conversation rename, LLM-generated auto-title ("A quieter week" style — avoids clinical terms in sidebar).

## [2026-04-13] — Phase A: Bahrain localization

- Crisis resources rewritten for Bahrain: **999** national emergency, **998** child protection, Salmaniya Psychiatric Hospital, Shamsaha 24/7 line. All US-centric numbers (988, 911, Crisis Text Line) removed.
- **CPR number** validation — 9-digit `YYMMNNNNC` format, month-plausibility check, display formatting (`8504-2345-6`), DOB extraction with century inference.
- **+973 phone** validation — auto-normalization of local / international / 00973 formats, mobile vs landline classification.
- **DD/MM/YYYY** dates throughout (`en-GB` locale, `Asia/Bahrain` timezone).
- LLM prompts tuned for a culture where mental-health stigma is a real barrier — non-judgmental, warm-but-substantive phrasing.

## [2026-04-13] — Phase Initial: Patient data model expansion

- New tables: Medication, Allergy, Diagnosis, ScreeningSchedule, Appointment, Notification, Conversation, CarePlan, AuditLog, EmailDelivery, KnowledgeChunk, PatientRAGChunk.
- User table extended with `date_of_birth`, `gender`, `cpr_number`, `nationality`, `phone`, `blood_type`, `profile_picture_url`, `language_preference`, `timezone`, `onboarding_completed`, social handles.
- Onboarding wizard endpoints.
- Direct patient ↔ clinician messaging.
- Recurring screening schedule with email reminders via Resend.

## [2026-04-13] — LLM: switch to Gemini 2.5 Flash

- Replaced DeepSeek R1 (402 Payment Required on OpenRouter free tier) with `google/gemini-2.5-flash:free`. Zero migration cost — same OpenAI-compatible client, one-line `.env` change. 256 K context, native Arabic, top-tier free-tier quality.
- `tenacity` retry wrapper on every LLM call.
- `max_tokens` reduced per endpoint (chat 400, verification 600, explanation 1200) to stay comfortably inside free-tier per-request limits.
