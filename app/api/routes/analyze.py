"""
Core screening API route.

Accepts text input, runs the full DSM-5 symptom detection pipeline
(DL classifier → LLM verification → RAG enrichment → LLM explanation),
persists results to DB, and ingests into patient RAG.

Requires authentication — screenings are always attributed to a patient.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import Screening, User, get_db
from app.schemas.analysis import (
    Evidence,
    ScreeningRequest,
    ScreeningResponse,
)
from app.services.auth import get_current_user, log_audit
from app.services.container import get_rag_service
from app.services.decision import DecisionService
from app.services.inference import ModelService
from app.services.llm import LLMService
from app.services.llm_verification import VerificationService
from app.services.rag import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instances (initialized on first request)
_model_service = None
_llm_service = None
_verification_service = None
_decision_service = None


async def get_services(settings: Settings = Depends(get_settings)):
    """Initialize and return all service instances."""
    global _model_service, _llm_service, _verification_service, _decision_service

    if _model_service is None:
        _model_service = ModelService(settings)
        await _model_service.load_models()
        _llm_service = LLMService(settings)
        _verification_service = VerificationService(settings)
        _decision_service = DecisionService()

    return {
        "model": _model_service,
        "llm": _llm_service,
        "verification": _verification_service,
        "decision": _decision_service,
        "rag": get_rag_service(),
        "settings": settings,
    }


@router.post("", response_model=ScreeningResponse)
@limiter.limit("20/minute")
async def screen_text(
    request: Request,
    body: ScreeningRequest,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Screen text for DSM-5 depression symptoms.

    Full pipeline:
    1. Sentence-level DL classification (DistilBERT → 11 classes)
    2. LLM verification (evidence validation, adversarial check, confidence calibration)
    3. Decision service (severity mapping, confidence adjustment, flagging)
    4. RAG retrieval (clinical context for detected symptoms)
    5. LLM explanation (per-symptom clinical narrative)
    6. Persist to DB + ingest into patient RAG

    Returns comprehensive screening result with sentence-level evidence.
    """
    model_service: ModelService = services["model"]
    llm_service: LLMService = services["llm"]
    verification_service: VerificationService = services["verification"]
    decision_service: DecisionService = services["decision"]
    rag_service: RAGService = services["rag"]
    settings: Settings = services["settings"]

    text = body.text
    screening_id = str(uuid4())
    created_at = datetime.now(UTC)

    logger.info(f"Screening {screening_id} started for patient {current_user.id[:8]} ({len(text)} chars)")

    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {settings.max_text_length} characters",
        )

    # Deduplication guard: reject if the same patient submitted another screening within 60 seconds
    recent_cutoff = datetime.now(UTC) - timedelta(seconds=60)
    recent = (
        db.query(Screening)
        .filter(
            Screening.patient_id == current_user.id,
            Screening.created_at >= recent_cutoff,
        )
        .first()
    )
    if recent:
        raise HTTPException(status_code=429, detail="Please wait before submitting another screening.")

    # Step 1: DL symptom detection
    symptom_analysis = await model_service.predict_symptoms(text)
    logger.info(
        f"  Step 1 — Detected {symptom_analysis.unique_symptom_count} symptoms, "
        f"severity={symptom_analysis.severity_level}"
    )

    # Steps 1.5 + 4: RAG retrieval (DSM-5 criteria + symptom context) in parallel
    async def _dsm5_retrieve() -> dict | None:
        if symptom_analysis.dsm5_criteria_met and rag_service and rag_service.is_initialized:

            def _fetch():
                ctx = {}
                for symptom in symptom_analysis.dsm5_criteria_met[:5]:
                    docs = rag_service.retrieve(
                        query=f"DSM-5 diagnostic criteria for {symptom}",
                        n_results=2,
                        category="dsm5_criteria",
                        symptom=symptom,
                    )
                    if docs:
                        ctx[symptom] = docs
                return ctx or None

            return await asyncio.to_thread(_fetch)
        return None

    async def _rag_retrieve() -> dict | None:
        if symptom_analysis.dsm5_criteria_met and rag_service and rag_service.is_initialized:
            return await asyncio.to_thread(
                rag_service.retrieve_for_symptoms,
                symptom_analysis.dsm5_criteria_met,
            )
        return None

    dsm5_context, rag_context_data = await asyncio.gather(
        _dsm5_retrieve(),
        _rag_retrieve(),
    )

    if dsm5_context:
        logger.info(f"  Step 1.5 — RAG: DSM-5 criteria for {len(dsm5_context)} symptoms")

    # Step 2: LLM verification (uses DSM-5 context from Step 1.5)
    verification = await verification_service.verify_prediction(
        text=text,
        symptom_analysis=symptom_analysis,
        dsm5_context=dsm5_context,
    )
    logger.info(
        f"  Step 2 — Verification: evidence_supports={verification.evidence_validation.evidence_supports_prediction}, "
        f"adversarial={verification.adversarial_check.likely_adversarial}, "
        f"trust={verification.confidence_analysis.should_trust_prediction}"
    )

    # Step 2.5: Filter false positives identified by LLM verification
    symptom_analysis = decision_service.filter_false_positives(symptom_analysis, verification)

    # Step 3: Decision (severity mapping + confidence adjustment)
    final_prediction, final_confidence, confidence_adjusted, flagged = decision_service.compute_final_prediction(
        symptom_analysis, verification
    )
    logger.info(f"  Step 3 — Decision: {final_prediction} ({final_confidence:.2%}), flagged={flagged}")

    # Step 4 (post-processing): format RAG context for LLM prompt
    rag_context_str = None
    if rag_context_data:
        rag_parts = []
        for symptom, docs in rag_context_data.items():
            for doc in docs:
                rag_parts.append(f"[{symptom}] {doc['text'][:800]}")
        rag_context_str = "\n\n".join(rag_parts[:8]) if rag_parts else None
        logger.info(f"  Step 4 — RAG: retrieved context for {len(rag_context_data)} symptoms")

    # Step 4.5: Assemble structured patient context
    patient_context_str = None
    try:
        from app.services.patient_context import PatientContextService

        pcs = PatientContextService()
        patient_context_str = pcs.build_context(
            current_user,
            db,
            sections=["demographics", "medications", "allergies", "diagnoses", "care_plan"],
            include_pii=False,
        )
    except Exception as e:
        logger.warning(f"Patient context assembly failed: {e}")

    # Step 5: LLM explanation
    verification_summary = decision_service.get_verification_summary(verification)
    explanation = await llm_service.generate_explanation(
        text=text,
        symptom_analysis=symptom_analysis,
        verification_summary=verification_summary,
        rag_context=rag_context_str,
        patient_context=patient_context_str,
    )
    logger.info("  Step 5 — Explanation generated")

    # Step 6a: Persist to DB
    adversarial_warning = None
    if verification.adversarial_check.likely_adversarial:
        adversarial_warning = (
            verification.adversarial_check.warning
            or f"Input flagged as potentially {verification.adversarial_check.adversarial_type}"
        )

    screening_record = Screening(
        id=screening_id,
        patient_id=current_user.id,
        created_at=created_at,
        text=text,
        source="api",
        symptom_data=symptom_analysis.model_dump(),
        symptom_count=symptom_analysis.unique_symptom_count,
        severity_level=symptom_analysis.severity_level,
        verification_data=verification.model_dump(),
        explanation_data=explanation.model_dump(),
        rag_context=rag_context_data,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        confidence_adjusted=confidence_adjusted,
        flagged_for_review=flagged,
        adversarial_warning=adversarial_warning,
    )
    db.add(screening_record)
    db.commit()

    # Send crisis alert email to the patient's clinician if flagged
    if flagged and current_user.clinician_id:
        try:
            from app.models.db import User as UserModel
            from app.services.email import get_email_service

            clinician = db.query(UserModel).filter(UserModel.id == current_user.clinician_id).first()
            if clinician and clinician.email:
                get_email_service(settings).send_crisis_alert_to_clinician(
                    clinician_name=clinician.full_name,
                    clinician_email=clinician.email,
                    patient_name=current_user.full_name,
                    severity=symptom_analysis.severity_level or "unknown",
                    symptom_count=symptom_analysis.unique_symptom_count or 0,
                    screening_id=screening_id,
                )
        except Exception as e:
            logger.warning(f"Crisis alert email failed: {e}")

    # Step 6b: Ingest into patient RAG for future chat context
    if rag_service and rag_service.is_initialized:
        rag_service.ingest_patient_screening(
            patient_id=current_user.id,
            screening_id=screening_id,
            text=text,
            symptoms_detected=[d.model_dump() for d in symptom_analysis.symptoms_detected],
            severity_level=symptom_analysis.severity_level,
        )

    # Audit log
    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    logger.info(f"Screening {screening_id} complete — saved to DB + RAG")

    # Build response
    evidence = Evidence(
        sentence_evidence=symptom_analysis.symptoms_detected,
        top_evidence_sentences=[
            d.sentence_text
            for d in sorted(
                symptom_analysis.symptoms_detected,
                key=lambda d: d.confidence,
                reverse=True,
            )[:5]
        ],
    )

    return ScreeningResponse(
        id=screening_id,
        created_at=created_at,
        text=text,
        symptom_analysis=symptom_analysis,
        evidence=evidence,
        verification=verification,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        confidence_adjusted=confidence_adjusted,
        explanation_report=explanation,
        flagged_for_review=flagged,
        adversarial_warning=adversarial_warning,
    )
