"""
Generate NEW training sentences directly from DSM-5 definitions.

Unlike paraphrasing (which rephrases existing sentences), this generates
original sentences from scratch — useful when existing samples are too few
or too specific for effective paraphrasing.

Targets: PSYCHOMOTOR, COGNITIVE_ISSUES (classes where paraphrase filter was too strict)

Usage:
    python augment_from_definitions.py --model gemini-3-flash-preview --count 40
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATION_PROMPTS = {
    "PSYCHOMOTOR": {
        "label_id": 4,
        "system": (
            "You are an expert in mental health text analysis. Generate realistic "
            "Reddit-style sentences where someone describes psychomotor changes "
            "as defined by DSM-5.\n\n"
            "Psychomotor symptoms include:\n"
            "- RETARDATION: Slowed physical movement, slow speech, taking long to respond, "
            "feeling physically heavy, body feels like lead, moving in slow motion\n"
            "- AGITATION: Restlessness, pacing, hand-wringing, can't sit still, "
            "fidgeting, nervous energy, feeling wired but exhausted\n\n"
            "IMPORTANT:\n"
            "- Use first-person, informal Reddit language\n"
            "- Focus on PHYSICAL/OBSERVABLE symptoms, not just feeling tired (that's FATIGUE)\n"
            "- Vary intensity: some mild ('I've been moving slower'), some severe ('I can barely move')\n"
            "- Include both retardation AND agitation examples\n"
            "- Each must be a single sentence\n"
            "- Return ONLY a JSON array of strings"
        ),
    },
    "COGNITIVE_ISSUES": {
        "label_id": 7,
        "system": (
            "You are an expert in mental health text analysis. Generate realistic "
            "Reddit-style sentences where someone describes cognitive difficulties "
            "as defined by DSM-5.\n\n"
            "Cognitive symptoms include:\n"
            "- Difficulty concentrating or focusing on tasks\n"
            "- Brain fog, mental cloudiness\n"
            "- Indecisiveness, difficulty making choices\n"
            "- Memory problems, forgetting things\n"
            "- Losing track of conversations or thoughts mid-sentence\n"
            "- Feeling mentally 'slow' or 'dull'\n"
            "- Unable to read or process information normally\n\n"
            "IMPORTANT:\n"
            "- Use first-person, informal Reddit language\n"
            "- Focus on COGNITIVE symptoms specifically (not fatigue, not sadness)\n"
            "- Vary the type: some about concentration, some about decisions, some about memory\n"
            "- Each must be a single sentence\n"
            "- Return ONLY a JSON array of strings"
        ),
    },
}


async def generate_sentences(
    client: AsyncOpenAI,
    symptom: str,
    count: int,
    model: str,
) -> list[str]:
    """Generate original sentences from DSM-5 definition."""
    config = GENERATION_PROMPTS[symptom]
    all_sentences = []
    batch_size = 15

    for _batch_start in range(0, count, batch_size):
        remaining = min(batch_size, count - len(all_sentences))

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": config["system"]},
                        {"role": "user", "content": f"Generate {remaining} unique sentences. JSON array only."},
                    ],
                    max_tokens=2048,
                    temperature=0.9,
                )

                content = response.choices[0].message.content
                if not content:
                    continue

                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    sentences = json.loads(match.group())
                    if isinstance(sentences, list):
                        all_sentences.extend([s for s in sentences if isinstance(s, str) and len(s) > 20])
                        break
            except Exception as e:
                logger.warning(f"  Attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2)

        await asyncio.sleep(1)

    return all_sentences[:count]


def filter_by_similarity(
    new_sentences: list[str],
    existing_sentences: list[str],
    min_sim: float = 0.55,
    max_sim: float = 0.95,
) -> list[tuple[str, float]]:
    """Filter generated sentences by similarity to existing training data."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    exist_emb = model.encode(existing_sentences, show_progress_bar=False)
    new_emb = model.encode(new_sentences, show_progress_bar=False)

    passed = []
    for _i, (sent, emb) in enumerate(zip(new_sentences, new_emb)):
        sims = np.dot(exist_emb, emb) / (
            np.linalg.norm(exist_emb, axis=1) * np.linalg.norm(emb)
        )
        max_sim_score = float(np.max(sims))

        if min_sim <= max_sim_score <= max_sim:
            passed.append((sent, max_sim_score))

    return passed


async def main():
    parser = argparse.ArgumentParser(description="Generate sentences from DSM-5 definitions")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--count", type=int, default=60, help="Sentences per class (PSYCHOMOTOR gets 2x)")
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--max-similarity", type=float, default=0.95)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "redsm5" / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")

    # Client
    api_key = ""
    env_path = base_dir.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("LLM_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    api_key = os.environ.get("LLM_API_KEY", api_key)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    all_new = []

    for symptom, config in GENERATION_PROMPTS.items():
        # PSYCHOMOTOR gets 2x volume because paraphrasing barely works for this class
        count = args.count * 2 if symptom == "PSYCHOMOTOR" else args.count
        logger.info(f"\n{'='*50}")
        logger.info(f"Generating {count} new {symptom} sentences from definition")

        # Generate
        sentences = await generate_sentences(client, symptom, count, args.model)
        logger.info(f"  Generated: {len(sentences)}")

        # Deduplicate
        sentences = list(set(sentences))
        logger.info(f"  After dedup: {len(sentences)}")

        # Filter by similarity to existing training data
        existing = train_df[train_df["label"] == symptom]["clean_text"].tolist()
        logger.info(f"  Existing {symptom} sentences: {len(existing)}")
        logger.info(f"  Filtering [{args.min_similarity}, {args.max_similarity}]...")

        passed = filter_by_similarity(
            sentences, existing,
            min_sim=args.min_similarity, max_sim=args.max_similarity,
        )
        logger.info(f"  Passed: {len(passed)}/{len(sentences)} ({len(passed)/max(len(sentences),1)*100:.0f}%)")

        for sent, sim in passed:
            all_new.append({
                "post_id": f"defgen_{symptom.lower()}_{len(all_new)}",
                "sentence_id": f"defgen_s_{symptom.lower()}_{len(all_new)}",
                "sentence_text": sent,
                "clean_text": sent,
                "label": symptom,
                "label_id": config["label_id"],
                "source": "definition_generated",
                "similarity_score": sim,
            })

    new_df = pd.DataFrame(all_new)
    new_df.to_csv(output_dir / "definition_generated.csv", index=False)

    # Merge with existing augmented data
    existing_aug_path = output_dir / "augmented_samples.csv"
    if existing_aug_path.exists():
        existing_aug = pd.read_csv(existing_aug_path)
        combined_aug = pd.concat([existing_aug, new_df], ignore_index=True)
        combined_aug.to_csv(output_dir / "augmented_samples_v2.csv", index=False)
        logger.info(f"\nMerged: {len(existing_aug)} existing + {len(new_df)} new = {len(combined_aug)} total augmented")

        # Also rebuild combined training set
        orig_df = train_df[["post_id", "sentence_id", "sentence_text", "clean_text", "label", "label_id"]].copy()
        orig_df["source"] = "original"
        orig_df["similarity_score"] = 1.0
        combined_train = pd.concat([orig_df, combined_aug[orig_df.columns]], ignore_index=True)
        combined_train = combined_train.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_train.to_csv(output_dir / "train_augmented_v2.csv", index=False)
        logger.info(f"Combined training set: {len(combined_train)} samples")
    else:
        new_df.to_csv(output_dir / "augmented_samples_v2.csv", index=False)

    print(f"\n{'='*60}")
    print("DEFINITION-BASED GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"New sentences generated: {len(new_df)}")
    for symptom in GENERATION_PROMPTS:
        count = len(new_df[new_df["label"] == symptom])
        existing_count = len(train_df[train_df["label"] == symptom])
        print(f"  {symptom}: {existing_count} original + {count} new = {existing_count + count}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
