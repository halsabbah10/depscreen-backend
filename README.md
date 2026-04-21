# DepScreen — Backend

FastAPI service for the DepScreen depression-screening platform. Runs the DSM-5 sentence classifier, LLM verification, RAG retrieval, chat, auth, and the clinician dashboard API. Deployed to HuggingFace Spaces (Docker SDK); this README doubles as the Space's homepage, which is why it carries the YAML frontmatter above.

Frontend (React + Vite): [halsabbah10/depscreen-frontend](https://github.com/halsabbah10/depscreen-frontend).

---

## What this service does

1. **Screens free-text input** (`POST /api/analyze`) through a 6-step pipeline:
   DistilBERT sentence classifier → 3-way LLM verification (evidence / adversarial / confidence) → decision layer (severity mapping, flagging, confidence adjustment) → pgvector RAG retrieval → LLM explanation → DB persist + patient-RAG ingest.
2. **Serves a RAG-grounded chat** (`/api/chat/*`) — standalone or linked to a specific screening, with crisis-keyword detection and streaming responses.
3. **Powers a clinician dashboard** (`/api/dashboard/*`) — patient triage, per-patient trends, care plans, appointments, direct notifications.
4. **Manages patient self-service** (`/api/patient/*`) — profile, medications, allergies, diagnoses, emergency contacts, onboarding, notifications, document uploads, PDF exports.
5. **Runs as a background-worker host** — APScheduler fires screening reminders, appointment reminders, and care-plan-review sweeps.

---

## Stack

| Concern | Choice |
|---|---|
| Framework | FastAPI 0.120 on Uvicorn |
| DB | Postgres (Supabase) + pgvector · SQLite in-memory for unit tests only (not supported for local dev) |
| Migrations | Alembic — idempotent boot-time sync via `init_db()` |
| ORM | SQLAlchemy 2.x |
| ML | PyTorch + Transformers — 3-model ensemble (DAPT DistilBERT + RoBERTa + DeBERTa, 0.813 micro-F1) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-d) |
| LLM | OpenAI SDK → Google AI Studio — Gemini Pro/Flash/Lite hybrid tiers |
| Resilience | `tenacity` retry with exponential backoff on every LLM call |
| Rate limiting | `slowapi` — auth 10/min, screening 20/min, chat 30/min |
| Logging | `structlog` JSON, request IDs, PII-scrubbed |
| Errors | Sentry (FastAPI integration) — no-op when DSN unset |
| Auth | JWT (python-jose) · bcrypt (passlib) · access 1h / refresh 7d |
| Email | Resend (via `resend` SDK); Jinja2 templates |
| PDF | Docling (parse uploaded forms, primary) · pdfplumber (fallback) · reportlab (generate reports) |
| Storage | Supabase Storage (profile avatars) |

---

## DSM-5 symptom classes

| # | Class | Code |
|---|---|---|
| 1 | Depressed mood | `DEPRESSED_MOOD` |
| 2 | Loss of interest / anhedonia | `ANHEDONIA` |
| 3 | Appetite / weight changes | `APPETITE_WEIGHT` |
| 4 | Sleep disturbance | `SLEEP_ISSUES` |
| 5 | Psychomotor changes | `PSYCHOMOTOR` |
| 6 | Fatigue / loss of energy | `FATIGUE` |
| 7 | Worthlessness / guilt | `WORTHLESSNESS_GUILT` |
| 8 | Concentration difficulties | `CONCENTRATION_ISSUES` |
| 9 | Suicidal ideation | `SUICIDAL_THOUGHTS` |
| 10 | Functional impairment | `FUNCTIONAL_IMPAIRMENT` |
| 11 | Non-depressive | `NON_DEPRESSIVE` |

Suicidal ideation always flags for clinician review, regardless of severity level or other signals.

---

## Safety layers

The screening pipeline has **four independent safety checks** that each LLM response passes through before reaching a patient:

1. **Crisis keyword scan** (chat layer) — deterministic substring match on suicidal / self-harm / resignation phrases triggers a Bahrain-localized crisis response.
2. **Adversarial detector** (verification layer) — catches prompt-injection, copypasta, gibberish, keyword-stuffing; flags input and drops confidence.
3. **Evidence validator** (verification layer) — LLM cross-checks whether the DL detections are actually supported by the sentence evidence.
4. **Safety guard** (output layer) — regex redaction of prescription advice, diagnostic claims, self-harm encouragement, and anti-professional content in LLM output. Replaces offending phrases with warm substitutes and appends a disclaimer with Bahrain crisis numbers.

Every clinically-significant action (screening submitted, crisis flagged, notification sent, care plan updated, auth event) writes to an immutable audit log.

---

## Running locally

### One-time setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in the values below
```

### Required environment variables

| Name | Purpose |
|---|---|
| `DATABASE_URL` | `postgresql://...` (Supabase or local Postgres with pgvector) — SQLite is not supported |
| `JWT_SECRET` | ≥32-char secret for signing access / refresh tokens |
| `LLM_API_KEY` | Google AI Studio key — leave blank to disable LLM calls (pipeline degrades gracefully) |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MODEL` | `gemini-3-flash-preview` (default) |
| `LLM_MODEL_PRO` | `gemini-3.1-pro-preview` (adversarial, explanation) |
| `LLM_MODEL_FLASH` | `gemini-3-flash-preview` (chat, evidence) |
| `LLM_MODEL_LITE` | `gemini-3.1-flash-lite-preview` (auto-title, utilities) |
| `CORS_ORIGINS` | JSON array, e.g. `["http://localhost:3000","https://depscreen.vercel.app"]` |

### Optional env vars

| Name | Purpose |
|---|---|
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | Avatar storage (falls back to Gravatar-style initials if unset) |
| `RESEND_API_KEY` / `RESEND_WEBHOOK_SECRET` | Transactional email + delivery webhooks |
| `SENTRY_DSN` | Error monitoring (no-op when empty) |
| `ENVIRONMENT` | `development` / `staging` / `production` |

### Boot

```bash
python -m uvicorn main:app --reload      # http://localhost:8000
# or
docker compose up --build                # full stack incl. Postgres
```

On startup the backend:
1. Enables the `vector` extension (no-op on SQLite)
2. Runs Alembic forward-only if `alembic_version` exists, otherwise `create_all` + stamp head
3. Warms the DistilBERT + sentence-transformer models
4. Starts APScheduler background jobs

Visit `http://localhost:8000/docs` for interactive Swagger once it's up.

---

## Endpoints

**Public:** `/api/auth/register`, `/api/auth/login`, `/api/auth/refresh`, `/health/live`, `/health/ready`

**Patient-scoped** (`/api/patient/*`, `/api/chat/*`, `/api/analyze`, `/api/history/*`, `/api/ingest/*`): screening, chat, medications, allergies, diagnoses, emergency contacts, screening schedule, notifications, documents, profile, avatar, onboarding, PDF export, Reddit/X ingestion, data export.

**Clinician-scoped** (`/api/dashboard/*`): patient list, patient detail, screenings, care plans, appointments, notifications, diagnoses, medications, schedule assignment, direct messaging, PDF summary.

**Shared:** `/api/terminology/*` (RxNorm drug autocomplete, ICD-10 code lookup), `/api/webhooks/*` (Resend delivery webhooks).

Full interactive API docs at `GET /docs` (OpenAPI 3.1).

---

## Migrations

```bash
# Generate a new migration from your model changes
alembic revision --autogenerate -m "descriptive message"

# Apply (or reapply) — idempotent
alembic upgrade head

# Roll back one revision
alembic downgrade -1
```

Live environments upgrade automatically on boot via `init_db()`. You never need to run Alembic by hand on HF Spaces unless you're debugging.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                   # 140 tests, ~20s
pytest --cov=app --cov-report=term       # with coverage (gate: 35%)
```

Tests use in-memory SQLite + mocked external services (LLM, RAG, email, Supabase). Rate limiting disabled, FastAPI lifespan replaced with no-op so tests don't load the 400 MB DistilBERT. Full notes in [`TESTING.md`](./TESTING.md).

CI runs the suite on every PR — see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

---

## Deployment (HuggingFace Spaces)

This repo deploys as-is to an HF Space on the **Docker SDK** (the YAML frontmatter at the top of this file configures it). Push to the Space's git remote and HF rebuilds the image automatically.

Space secrets (set in the HF dashboard, never in the repo):

```
DATABASE_URL, JWT_SECRET, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL,
CORS_ORIGINS, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
RESEND_API_KEY, RESEND_WEBHOOK_SECRET, SENTRY_DSN, ENVIRONMENT=production
```

### Readiness checks

`/health/live` — is the process up?
`/health/ready` — DB + symptom classifier + RAG + LLM config all green?

HF Spaces load balancer uses `/health/live`; monitoring should scrape `/health/ready`.

---

## Project layout

```
backend/
├── app/
│   ├── api/routes/         # 9 routers: auth, analyze, chat, dashboard,
│   │                         history, ingest, patient, terminology, webhooks
│   ├── services/           # llm, rag, chat, inference, decision,
│   │                         llm_verification, safety_guard, email,
│   │                         avatar, scheduler, patient_context, …
│   ├── models/db.py        # SQLAlchemy models (17 tables)
│   ├── schemas/analysis.py # All Pydantic request/response shapes
│   ├── middleware/         # error handler, rate limiter, request logging,
│   │                         security headers, LLM resilience
│   └── core/               # config, sentry, Bahrain localization helpers
├── alembic/                # Migrations
├── ml/
│   ├── scripts/            # Training pipeline
│   ├── models/             # Trained DistilBERT weights
│   ├── knowledge_base/     # 23 clinical RAG documents
│   └── evaluation/         # Metrics, ablations, confusion matrices
├── tests/                  # pytest suite (140 tests)
├── Dockerfile              # Used by HF Spaces
├── main.py                 # FastAPI entrypoint
└── README.md               # ← you are here
```

---

## Not for clinical use

DepScreen is a **screening aid, not a diagnostic tool.** No BAA with the LLM provider, no FHIR integration, no regulatory certification. Crisis resources are labeled everywhere — the app is designed to surface professional help, not replace it.

If you are in Bahrain and in crisis: **999** (national emergency) or **Shamsaha 17651421** (24/7).
