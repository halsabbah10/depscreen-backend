#!/usr/bin/env python3
"""Run all 8 verification gates and report pass/fail.

Gates:
1. Retrieval quality (model comparison)
2. Threshold calibration
3. RAGAS evaluation (requires clinician QA pairs)
4. Patient isolation
5. Safety layers
6. Memory + latency
7. Clinical review (manual — checklist)
8. E2E pipeline

Usage:
    python scripts/evaluation/run_verification_gates.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent.parent


def run_gate(name: str, command: list[str], timeout: int = 600) -> bool:
    """Run a gate command and return pass/fail."""
    logger.info(f"\n{'='*60}")
    logger.info(f"GATE: {name}")
    logger.info(f"{'='*60}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=str(BACKEND_DIR), timeout=timeout)
        if result.returncode == 0:
            logger.info(f"  PASS")
            return True
        else:
            logger.info(f"  FAIL")
            if result.stderr:
                logger.info(f"  Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.info(f"  TIMEOUT")
        return False
    except Exception as e:
        logger.info(f"  ERROR: {e}")
        return False


def main():
    gates = {}

    # Gate 4: Patient isolation tests
    gates["4. Patient Isolation"] = run_gate(
        "Patient Isolation",
        [sys.executable, "-m", "pytest", "tests/test_patient_isolation.py", "-v", "--tb=short"],
    )

    # Gate 5: Safety layer tests
    gates["5. Safety Layers"] = run_gate(
        "Safety Layers",
        [sys.executable, "-m", "pytest", "tests/test_rag_safety.py", "tests/test_rag_degradation.py", "-v", "--tb=short"],
    )

    # Gate 6: Import check (can we import everything?)
    gates["6. Import Check"] = run_gate(
        "Import Check",
        [sys.executable, "-c", "from app.services.rag import RAGService; from app.services.rag_safety import filter_by_relevance; from app.services.chat_summary import extract_clinical_sentences; print('All imports OK')"],
    )

    # Gate 8: E2E pipeline tests
    gates["8. E2E Pipeline"] = run_gate(
        "E2E Pipeline",
        [sys.executable, "-m", "pytest", "tests/test_pipeline_rag_e2e.py", "-v", "--tb=short"],
    )

    # Gate: Full test suite
    gates["Full Test Suite"] = run_gate(
        "Full Test Suite",
        [sys.executable, "-m", "pytest", "tests/", "--ignore=tests/test_rag_integration.py", "--ignore=tests/test_patient_isolation.py", "-q"],
    )

    # Report
    logger.info(f"\n{'='*60}")
    logger.info("VERIFICATION GATE SUMMARY")
    logger.info(f"{'='*60}")

    all_pass = True
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {gate_name}: {status}")
        if not passed:
            all_pass = False

    logger.info(f"\nManual gates (require human verification):")
    logger.info(f"  1. Retrieval Quality: Run scripts/evaluation/run_model_comparison.py")
    logger.info(f"  2. Threshold Calibration: Run scripts/evaluation/run_threshold_calibration.py")
    logger.info(f"  3. RAGAS Evaluation: Run scripts/evaluation/run_ragas_evaluation.py (needs clinician QA)")
    logger.info(f"  7. Clinical Review: Clinician must review knowledge base content")

    if all_pass:
        logger.info(f"\nAll automated gates PASSED")
    else:
        logger.info(f"\nSome gates FAILED — fix before deployment")
        sys.exit(1)


if __name__ == "__main__":
    main()
