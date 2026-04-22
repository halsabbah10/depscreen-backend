#!/usr/bin/env python3
"""Calibrate the relevance threshold for the reranker.

Tests retrieval at various threshold values and reports precision/recall
to find the optimal F1 threshold. Used for Gate 2.

Usage:
    python scripts/evaluation/run_threshold_calibration.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    eval_path = Path(__file__).parent.parent.parent / "tests" / "evaluation" / "eval_queries.json"
    queries = json.loads(eval_path.read_text())
    relevant_queries = [q for q in queries if q.get("expected_category") is not None]

    logger.info(f"Loaded {len(relevant_queries)} relevant queries for calibration")

    import asyncio

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.core.config import get_settings
    from app.services.rag import RAGService

    settings = get_settings()
    rag = RAGService(settings)
    asyncio.run(rag.initialize())

    # Test thresholds from 0.20 to 0.60
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    logger.info("\nThreshold | Precision | Recall | F1")
    logger.info("-" * 45)

    best_f1 = 0
    best_threshold = 0.40

    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for q in relevant_queries:
            results = rag.retrieve(q["query"], n_results=10)
            if not results:
                false_negatives += 1
                continue

            # Filter by threshold
            from app.services.rag_safety import filter_by_relevance

            filtered = filter_by_relevance(results, threshold=threshold)

            if not filtered:
                false_negatives += 1
            else:
                top_cat = filtered[0].get("metadata", {}).get("category", "")
                if top_cat == q["expected_category"]:
                    true_positives += 1
                else:
                    false_positives += 1

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        logger.info(f"  {threshold:.2f}    |  {precision:.3f}    | {recall:.3f}  | {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"\nOptimal threshold: {best_threshold} (F1={best_f1:.3f})")
    logger.info(f"Update config: rag_similarity_threshold = {best_threshold}")

    output = {"thresholds_tested": thresholds, "best_threshold": best_threshold, "best_f1": best_f1}
    output_path = Path(__file__).parent / "threshold_calibration_results.json"
    output_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
