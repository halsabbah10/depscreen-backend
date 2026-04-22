#!/usr/bin/env python3
"""Compare embedding models on domain-specific retrieval queries.

Runs the evaluation query set against the loaded embedding model and
reports retrieval quality metrics. Used to validate model choice
before deployment (Gate 1).

Usage:
    python scripts/evaluation/run_model_comparison.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Load evaluation queries
    eval_path = Path(__file__).parent.parent.parent / "tests" / "evaluation" / "eval_queries.json"
    if not eval_path.exists():
        logger.error(f"Evaluation queries not found: {eval_path}")
        sys.exit(1)

    queries = json.loads(eval_path.read_text())
    logger.info(f"Loaded {len(queries)} evaluation queries")

    # Initialize RAG service
    try:
        import asyncio
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.core.config import get_settings
        from app.services.rag import RAGService

        settings = get_settings()
        rag = RAGService(settings)
        asyncio.run(rag.initialize())
        logger.info(f"RAG service initialized with model: {settings.rag_embedding_model}")
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        sys.exit(1)

    # Run queries
    results = []
    correct_category = 0
    total_with_results = 0
    total_relevant = 0  # queries that should have results

    for q in queries:
        query_text = q["query"]
        expected_cat = q.get("expected_category")

        retrieved = rag.retrieve(query_text, n_results=5)

        if expected_cat is None:
            # Irrelevant query — should return empty or low-relevance results
            if not retrieved or len(retrieved) == 0:
                results.append({"query": query_text, "status": "CORRECT_EMPTY", "expected": None})
            else:
                results.append({"query": query_text, "status": "FALSE_POSITIVE", "top_category": retrieved[0].get("metadata", {}).get("category")})
            continue

        total_relevant += 1

        if not retrieved:
            results.append({"query": query_text, "status": "NO_RESULTS", "expected": expected_cat})
            continue

        total_with_results += 1
        top_cat = retrieved[0].get("metadata", {}).get("category", "")

        if top_cat == expected_cat:
            correct_category += 1
            results.append({"query": query_text, "status": "CORRECT", "top_category": top_cat})
        else:
            results.append({"query": query_text, "status": "WRONG_CATEGORY", "expected": expected_cat, "got": top_cat})

    # Report
    logger.info("\n=== MODEL COMPARISON RESULTS ===")
    logger.info(f"Model: {settings.rag_embedding_model}")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Relevant queries: {total_relevant}")
    logger.info(f"Queries with results: {total_with_results}")
    logger.info(f"Correct top category: {correct_category}/{total_relevant} ({correct_category/max(total_relevant,1)*100:.1f}%)")

    # Show failures
    failures = [r for r in results if r["status"] in ("WRONG_CATEGORY", "NO_RESULTS")]
    if failures:
        logger.info(f"\nFailures ({len(failures)}):")
        for f in failures:
            logger.info(f"  [{f['status']}] {f['query'][:60]}... expected={f.get('expected')}, got={f.get('got', 'N/A')}")

    # Save results
    output_path = Path(__file__).parent / "model_comparison_results.json"
    output_path.write_text(json.dumps({"model": settings.rag_embedding_model, "results": results}, indent=2))
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
