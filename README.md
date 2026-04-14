---
title: DepScreen
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# DepScreen Backend

AI-powered depression screening platform with DSM-5 symptom detection, LLM verification, and RAG clinical context.

## Architecture

- **ML Model**: Fine-tuned DistilBERT for sentence-level DSM-5 symptom classification (0.696 micro-F1)
- **LLM**: Gemini 2.5 Flash for clinical explanation and verification
- **RAG**: pgvector-powered clinical knowledge base (23 documents)
- **Database**: Supabase PostgreSQL with pgvector extension
- **Auth**: JWT with RBAC (patient/clinician/admin roles)
- **API**: FastAPI with 72 endpoints across 7 route modules

## DSM-5 Symptom Classes

1. Depressed mood
2. Loss of interest (anhedonia)
3. Appetite/weight changes
4. Sleep disturbance
5. Psychomotor changes
6. Fatigue/loss of energy
7. Worthlessness/guilt
8. Concentration difficulties
9. Suicidal ideation
10. Functional impairment
11. Non-depressive

## Tech Stack

- FastAPI + Uvicorn
- PyTorch + Transformers (DistilBERT)
- sentence-transformers (all-MiniLM-L6-v2)
- SQLAlchemy + Alembic + pgvector
- OpenAI SDK (Gemini-compatible)
- slowapi (rate limiting)
- tenacity (LLM retry)
- structlog (structured logging)
