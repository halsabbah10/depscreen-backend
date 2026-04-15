# Backend Testing

## What's covered

| Layer | Files | Count | Strategy |
|---|---|---|---|
| Auth routes | `tests/test_auth.py` | 16 | real HTTP, real SQLite, bcrypt + JWT round-trips |
| Screening route | `tests/test_screening.py` | 9 | real HTTP, services mocked at `Depends(get_services)` |
| Chat CRUD | `tests/test_chat.py` | 10 | real HTTP, no LLM (streaming covered in E2E) |
| Patient CRUD | `tests/test_patient.py` | 14 | real HTTP, patient-scoped data isolation checks |
| Dashboard | `tests/test_dashboard.py` | 11 | exercises Batch-2 N+1 refactor, RBAC, notify endpoint |
| DecisionService | `tests/test_decision.py` | 19 | pure unit, every confidence + flagging branch |
| SafetyGuard | `tests/test_safety_guard.py` | 24 | prescription / diagnosis / self-harm / undermine categories |
| Localization + crisis | `tests/test_services.py` | 37 | CPR, +973 phone, DD/MM/YYYY, crisis keyword lexicon |
| **Total** | | **140** | |

Coverage: **36%** global, **70–95%** on routes we test. The heavy service
layers (LLM, RAG, inference, PDF, reports) are mocked at the route
boundary — they're exercised for real via Playwright E2E in
`../frontend/e2e/`.

## Running

```bash
# Install test deps (once)
pip install -r requirements-dev.txt

# Full suite with coverage
pytest tests/ --cov=app --cov-report=term

# Single file
pytest tests/test_auth.py -v

# Stop at first failure
pytest tests/ -x
```

## Environment

Tests use `sqlite:///:memory:` — no external DB needed. External services
(LLM, Resend, Supabase Storage) are mocked at module import time; a test
run never reaches the network. If you need to override any env:

```bash
DATABASE_URL="postgresql://..." pytest tests/
```

## Design notes

- **In-memory SQLite.** The two pgvector-backed tables (`knowledge_chunks`,
  `patient_rag_chunks`) are skipped during `create_all()` — tests that
  need RAG mock the service layer, not the storage.
- **Rate limiting disabled in tests.** `app.state.limiter.enabled = False`
  — the IP-based limiter would otherwise throttle parallel runs.
- **FastAPI lifespan replaced with no-op.** We don't load the 400 MB
  DistilBERT or start the APScheduler during collect/run.
- **Truncation, not rollback, for isolation.** Route handlers commit,
  so rolling back the fixture session would be a no-op. Instead every
  test ends by truncating each table.

## CI

Every PR runs the full suite via `.github/workflows/ci.yml` (`tests` job).
Fails the pipeline on <35% coverage.
