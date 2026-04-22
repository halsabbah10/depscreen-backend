#!/usr/bin/env python3
"""Download Tier 1 source PDFs for the clinical knowledge base.

Run to populate ml/knowledge_base/sources/ with authoritative clinical
documents from public sources. Idempotent — existing files skipped.

Usage:
    python scripts/fetch_knowledge_base.py
    python scripts/fetch_knowledge_base.py --force  # Re-download all
"""

import logging
import sys
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCES_DIR = Path(__file__).parent.parent / "ml" / "knowledge_base" / "sources"

DOCUMENTS = [
    {
        "filename": "phq9_questionnaire_apa.pdf",
        "url": "https://www.apa.org/depression-guideline/patient-health-questionnaire.pdf",
        "description": "PHQ-9 Questionnaire (APA) — public domain",
    },
    {
        "filename": "gad7_questionnaire_adaa.pdf",
        "url": "https://adaa.org/sites/default/files/GAD-7_Anxiety-updated_0.pdf",
        "description": "GAD-7 Questionnaire (ADAA) — public domain",
    },
    {
        "filename": "cssrs_baseline_screening.pdf",
        "url": "https://cssrs.columbia.edu/wp-content/uploads/C-SSRS1-14-09-BaselineScreening.pdf",
        "description": "C-SSRS Baseline/Screening Version — free for healthcare",
    },
    {
        "filename": "cssrs_scoring_guide.pdf",
        "url": "https://cssrs.columbia.edu/wp-content/uploads/ScoringandDataAnalysisGuide-for-Clinical-Trials-1.pdf",
        "description": "C-SSRS Scoring Guide — free for healthcare",
    },
    {
        "filename": "apa_depression_guideline_2019.pdf",
        "url": "https://www.apa.org/depression-guideline/guideline.pdf",
        "description": "APA Clinical Practice Guideline for Depression (2019)",
    },
    {
        "filename": "apa_mdd_practice_guideline.pdf",
        "url": "https://psychiatryonline.org/pb/assets/raw/sitewide/practice_guidelines/guidelines/mdd.pdf",
        "description": "APA Practice Guideline for MDD",
    },
    {
        "filename": "nice_cg91_depression_chronic_health.pdf",
        "url": "https://www.nice.org.uk/guidance/cg91/resources/depression-in-adults-with-a-chronic-physical-health-problem-recognition-and-management-pdf-975744316357",
        "description": "NICE CG91: Depression with chronic health problems",
    },
    {
        "filename": "dsm5tr_mdd_fact_sheet.pdf",
        "url": "https://www.psychiatry.org/getmedia/33fc7cdb-6fd8-46a7-9ff2-225ba7862f7f/APA-DSM5TR-MajorDepressiveDisorder.pdf",
        "description": "DSM-5-TR MDD Fact Sheet",
    },
    {
        "filename": "esketamine_fda_label_2025.pdf",
        "url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/211243s016lbl.pdf",
        "description": "Esketamine (Spravato) FDA Label — public domain",
    },
    {
        "filename": "who_bahrain_mental_health.pdf",
        "url": "https://cdn.who.int/media/docs/default-source/mental-health/who-aims-country-reports/mh_aims_report_bahrain_jan_2011_en.pdf",
        "description": "WHO-AIMS: Mental Health System in Bahrain",
    },
]


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file. Returns True if downloaded, False if skipped/failed."""
    if dest.exists():
        logger.info(f"  SKIP (exists): {dest.name}")
        return False

    logger.info(f"  Downloading: {description}")
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
            size_mb = len(response.content) / (1024 * 1024)
            logger.info(f"  OK: {dest.name} ({size_mb:.1f} MB)")
            return True
    except httpx.HTTPError as e:
        logger.error(f"  FAILED: {dest.name} — {e}")
        logger.error(f"  → Manual download needed: {url}")
        return False


def main():
    force = "--force" in sys.argv
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching {len(DOCUMENTS)} source PDFs to {SOURCES_DIR}")
    downloaded, skipped, failed = 0, 0, 0
    failed_docs = []

    for doc in DOCUMENTS:
        dest = SOURCES_DIR / doc["filename"]
        if force and dest.exists():
            dest.unlink()

        success = download_file(doc["url"], dest, doc["description"])
        if success:
            downloaded += 1
        elif dest.exists():
            skipped += 1
        else:
            failed += 1
            failed_docs.append(doc)

    logger.info(f"\nSummary: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    if failed_docs:
        logger.warning("\nFailed downloads (manual fetch needed):")
        for doc in failed_docs:
            logger.warning(f"  {doc['filename']}: {doc['url']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
