"""
Data augmentation for rare classes via LLM paraphrasing.

For each sentence in an underrepresented class, generates N paraphrases
using Gemini Flash, then filters by semantic similarity to ensure
quality without drift.

Target classes (from training data):
  PSYCHOMOTOR:     30 → 150
  APPETITE_CHANGE: 41 → 150
  COGNITIVE_ISSUES: 51 → 150
  SPECIAL_CASE:    76 → 150
  SUICIDAL_THOUGHTS: +50 indirect phrasings

Usage:
    python augment_rare_classes.py [options]

    # Run in terminal with progress:
    cd backend && source venv/bin/activate
    python ml/scripts/augment_rare_classes.py --model gemini-3-flash-preview --delay 0.5
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DSM-5 definitions for grounding paraphrases
DSM5_DEFINITIONS = {
    "PSYCHOMOTOR": (
        "Observable slowing of physical movement, speech, or thought (retardation), "
        "OR physical restlessness and agitation (pacing, hand-wringing, inability to sit still). "
        "Must be observable by others."
    ),
    "APPETITE_CHANGE": (
        "Significant change in appetite or weight (increase or decrease) "
        "not due to intentional dieting. Food losing its appeal, "
        "eating significantly more or less than usual, noticeable weight change."
    ),
    "COGNITIVE_ISSUES": (
        "Diminished ability to think, concentrate, or make decisions. "
        "Brain fog, difficulty focusing, indecisiveness, memory problems. "
        "Inability to complete tasks, losing track of conversations."
    ),
    "SPECIAL_CASE": (
        "Clinical concern related to depression that doesn't map cleanly "
        "to the 9 standard DSM-5 criteria. General distress, social withdrawal, "
        "or mixed symptom presentation."
    ),
    "SUICIDAL_THOUGHTS": (
        "Recurrent thoughts of death, suicidal ideation, or self-harm. "
        "Ranges from passive ('I wish I didn't exist', 'everyone would be better off without me') "
        "to active ('I want to kill myself'). Includes indirect expressions."
    ),
}

PARAPHRASE_SYSTEM_PROMPT = """\
You are an expert in mental health text analysis. Your task is to create \
realistic paraphrases of Reddit-style mental health posts that preserve \
the clinical symptom while varying language, tone, and sentence structure.

RULES:
1. Each paraphrase must express the SAME DSM-5 symptom as the original.
2. Use informal Reddit-style language (first person, conversational, may include slang).
3. Vary: vocabulary, sentence length, emotional intensity, level of directness.
4. Do NOT add symptoms that aren't in the original.
5. Do NOT make the text sound clinical or formal — keep it natural.
6. Each paraphrase should be a single sentence (not a paragraph).
7. Return ONLY a JSON array of strings. No explanation.
"""

INDIRECT_SUICIDAL_SYSTEM_PROMPT = """\
You are an expert in mental health text analysis. Your task is to create \
realistic examples of INDIRECT or PASSIVE suicidal ideation as expressed \
in Reddit mental health posts.

These are sentences where the person does NOT explicitly say "I want to die" \
or "I want to kill myself" but instead expresses:
- Wishing they didn't exist ("I wish I could just disappear")
- Being a burden ("Everyone would be better off without me")
- Exhaustion with life ("I'm so tired of fighting every day")
- Passive death wishes ("I wouldn't mind if I didn't wake up")
- Nihilistic hopelessness ("What's even the point of going on")

RULES:
1. Each sentence must be recognizable as passive suicidal ideation to a clinician.
2. Use informal Reddit-style language.
3. Vary intensity, directness, and phrasing.
4. Return ONLY a JSON array of strings. No explanation.
"""


async def generate_paraphrases(
    client: AsyncOpenAI,
    sentence: str,
    symptom: str,
    n: int = 5,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> list[str]:
    """Generate N paraphrases of a sentence preserving the symptom."""
    definition = DSM5_DEFINITIONS.get(symptom, "")

    user_prompt = (
        f"Original sentence (symptom: {symptom}):\n"
        f'"{sentence}"\n\n'
        f"DSM-5 definition of {symptom}: {definition}\n\n"
        f"Generate {n} paraphrases as a JSON array of {n} strings."
    )

    for _attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.8,  # Higher temp for variety
            )

            content = response.choices[0].message.content
            if not content:
                continue

            # Extract JSON array
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                paraphrases = json.loads(match.group())
                if isinstance(paraphrases, list) and all(isinstance(p, str) for p in paraphrases):
                    return paraphrases[:n]

        except Exception as e:
            if "429" in str(e):
                await asyncio.sleep(30)
            else:
                await asyncio.sleep(1)

    return []


async def generate_indirect_suicidal(
    client: AsyncOpenAI,
    n: int = 50,
    model: str = "gemini-3-flash-preview",
) -> list[str]:
    """Generate indirect/passive suicidal ideation examples."""
    all_examples = []
    batch_size = 10

    for _batch_start in range(0, n, batch_size):
        remaining = min(batch_size, n - len(all_examples))

        user_prompt = (
            f"Generate {remaining} unique examples of indirect/passive suicidal ideation "
            f"as Reddit mental health post sentences. JSON array of strings only."
        )

        for _attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": INDIRECT_SUICIDAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1024,
                    temperature=0.9,
                )

                content = response.choices[0].message.content
                if not content:
                    continue

                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    examples = json.loads(match.group())
                    if isinstance(examples, list):
                        all_examples.extend([e for e in examples if isinstance(e, str)])
                        break
            except Exception:
                await asyncio.sleep(2)

        await asyncio.sleep(1)

    return all_examples[:n]


def compute_similarity_filter(
    originals: list[str],
    paraphrases: list[str],
    min_sim: float = 0.70,
    max_sim: float = 0.95,
) -> list[tuple[str, float]]:
    """Filter paraphrases by semantic similarity to originals.

    Keeps only paraphrases with similarity in [min_sim, max_sim]:
    - Below min_sim: drifted too far from original meaning
    - Above max_sim: too close to original (near-duplicate)

    Returns list of (paraphrase, similarity_score) tuples that passed.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    orig_embeddings = model.encode(originals, show_progress_bar=False)
    para_embeddings = model.encode(paraphrases, show_progress_bar=False)

    passed = []
    for _i, (para, para_emb) in enumerate(zip(paraphrases, para_embeddings)):
        # Find max similarity to any original
        sims = np.dot(orig_embeddings, para_emb) / (
            np.linalg.norm(orig_embeddings, axis=1) * np.linalg.norm(para_emb)
        )
        max_sim_score = float(np.max(sims))

        if min_sim <= max_sim_score <= max_sim:
            passed.append((para, max_sim_score))

    return passed


async def main():
    parser = argparse.ArgumentParser(description="Augment rare classes via LLM paraphrasing")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--paraphrases-per-sentence", type=int, default=5)
    parser.add_argument("--target-per-class", type=int, default=150)
    parser.add_argument("--indirect-suicidal-count", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-similarity", type=float, default=0.70)
    parser.add_argument("--max-similarity", type=float, default=0.95)
    parser.add_argument("--psychomotor-min-sim", type=float, default=0.50,
                        help="Lower similarity threshold for PSYCHOMOTOR (default: 0.50)")
    parser.add_argument("--cognitive-min-sim", type=float, default=0.55,
                        help="Lower similarity threshold for COGNITIVE_ISSUES (default: 0.55)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "redsm5" / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_csv(data_dir / "train.csv")
    logger.info(f"Loaded {len(train_df)} training samples")

    # Identify rare classes
    class_counts = train_df["label"].value_counts()
    target_classes = {
        cls: args.target_per_class - count
        for cls, count in class_counts.items()
        if count < args.target_per_class and cls in DSM5_DEFINITIONS
    }

    logger.info("\nAugmentation targets:")
    for cls, needed in sorted(target_classes.items(), key=lambda x: x[1], reverse=True):
        current = class_counts[cls]
        logger.info(f"  {cls}: {current} → {args.target_per_class} (need {needed} more)")

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

    # ── Per-class similarity thresholds ──
    CLASS_MIN_SIM = {
        "PSYCHOMOTOR": args.psychomotor_min_sim,      # 0.50 — physical symptoms are semantically diverse
        "COGNITIVE_ISSUES": args.cognitive_min_sim,    # 0.55 — cognitive symptoms vary widely
    }

    # Per-class paraphrase multiplier (generate more for rarest classes)
    CLASS_VOLUME_MULTIPLIER = {
        "PSYCHOMOTOR": 3,       # 27 sources → need 3x volume to overcome filter
        "COGNITIVE_ISSUES": 2,  # 43 sources → need 2x
    }

    # ── Generate paraphrases for each rare class ──
    all_augmented = []
    semaphore = asyncio.Semaphore(args.concurrency)

    for cls, needed in target_classes.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Augmenting {cls} (need {needed} more)")

        # Get existing sentences for this class
        class_sentences = train_df[train_df["label"] == cls]["clean_text"].tolist()
        originals_for_filter = list(class_sentences)

        # Per-class volume: generate more for harder classes
        multiplier = CLASS_VOLUME_MULTIPLIER.get(cls, 1)
        n_per = max(args.paraphrases_per_sentence * multiplier, (needed * 3 // len(class_sentences)) + 1)
        logger.info(f"  {len(class_sentences)} source sentences × {n_per} paraphrases each (multiplier={multiplier}x)")

        # Per-class similarity threshold
        min_sim = CLASS_MIN_SIM.get(cls, args.min_similarity)
        logger.info(f"  Similarity threshold: [{min_sim}, {args.max_similarity}]")

        raw_paraphrases = []
        done = 0
        total = len(class_sentences)

        async def gen_one(sentence: str, *, _semaphore=semaphore, _cls=cls, _n_per=n_per, _total=total):
            nonlocal done
            async with _semaphore:
                result = await generate_paraphrases(
                    client, sentence, _cls, n=_n_per, model=args.model,
                )
                await asyncio.sleep(args.delay)
                done += 1
                pct = done / _total * 100
                bar_len = 30
                filled = int(bar_len * done / _total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {done}/{_total} ({pct:.0f}%)", end="", flush=True)
                return result

        tasks = [gen_one(s) for s in class_sentences]
        results = await asyncio.gather(*tasks)
        print()

        for paras in results:
            raw_paraphrases.extend(paras)

        logger.info(f"  Generated {len(raw_paraphrases)} raw paraphrases")

        # Deduplicate
        raw_paraphrases = list(set(raw_paraphrases))
        logger.info(f"  After dedup: {len(raw_paraphrases)}")

        # Similarity filter with per-class threshold
        logger.info(f"  Filtering by similarity [{min_sim}, {args.max_similarity}]...")
        passed = compute_similarity_filter(
            originals_for_filter, raw_paraphrases,
            min_sim=min_sim, max_sim=args.max_similarity,
        )
        logger.info(f"  Passed filter: {len(passed)}/{len(raw_paraphrases)} ({len(passed)/max(len(raw_paraphrases),1)*100:.0f}%)")

        # Take only what we need
        selected = passed[:needed]
        logger.info(f"  Selected: {len(selected)} (target was {needed})")

        for para, sim in selected:
            all_augmented.append({
                "clean_text": para,
                "label": cls,
                "label_id": train_df[train_df["label"] == cls]["label_id"].iloc[0],
                "source": "augmented",
                "similarity_score": sim,
            })

    # ── Definition-based generation for classes where paraphrasing underperforms ──
    DEFINITION_TARGETS = {
        "PSYCHOMOTOR": 60,
        "COGNITIVE_ISSUES": 40,
    }

    for def_cls, def_count in DEFINITION_TARGETS.items():
        # Check how many we already got from paraphrasing
        already = len([a for a in all_augmented if a["label"] == def_cls])
        target = target_classes.get(def_cls, 0)
        still_needed = target - already
        if still_needed <= 0:
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Definition-based generation for {def_cls} (paraphrasing got {already}, still need {still_needed})")

        definition = DSM5_DEFINITIONS.get(def_cls, "")
        gen_prompt = (
            f"Generate {def_count} unique Reddit-style first-person sentences where someone "
            f"describes {def_cls.replace('_', ' ').lower()} symptoms.\n\n"
            f"DSM-5 definition: {definition}\n\n"
            f"Rules:\n"
            f"- Informal Reddit language, first person\n"
            f"- Each must be a single sentence\n"
            f"- Vary intensity, vocabulary, and directness\n"
            f"- Return ONLY a JSON array of strings"
        )

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                        {"role": "user", "content": gen_prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.9,
                )
                content = response.choices[0].message.content
                if content:
                    content = re.sub(r"```json\s*", "", content)
                    content = re.sub(r"```\s*$", "", content)
                    match = re.search(r"\[.*\]", content, re.DOTALL)
                    if match:
                        def_sentences = json.loads(match.group())
                        def_sentences = [s for s in def_sentences if isinstance(s, str) and len(s) > 20]
                        def_sentences = list(set(def_sentences))

                        # Filter with class-specific threshold
                        cls_min_sim = CLASS_MIN_SIM.get(def_cls, args.min_similarity)
                        existing = train_df[train_df["label"] == def_cls]["clean_text"].tolist()
                        passed_def = compute_similarity_filter(
                            existing, def_sentences,
                            min_sim=cls_min_sim, max_sim=args.max_similarity,
                        )
                        logger.info(f"  Generated {len(def_sentences)}, passed filter: {len(passed_def)}")

                        label_id = int(train_df[train_df["label"] == def_cls]["label_id"].iloc[0])
                        for para, sim in passed_def[:still_needed]:
                            all_augmented.append({
                                "clean_text": para,
                                "label": def_cls,
                                "label_id": label_id,
                                "source": "definition_generated",
                                "similarity_score": sim,
                            })
                        break
            except Exception as e:
                logger.warning(f"  Definition generation attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2)

        final_count = len([a for a in all_augmented if a["label"] == def_cls])
        logger.info(f"  {def_cls} total augmented: {final_count}")

    # ── Generate indirect suicidal ideation ──
    logger.info(f"\n{'='*50}")
    logger.info(f"Generating {args.indirect_suicidal_count} indirect suicidal ideation examples")

    indirect_examples = await generate_indirect_suicidal(
        client, n=args.indirect_suicidal_count, model=args.model,
    )

    # Filter against existing suicidal_thoughts sentences
    existing_suicidal = train_df[train_df["label"] == "SUICIDAL_THOUGHTS"]["clean_text"].tolist()
    passed_indirect = compute_similarity_filter(
        existing_suicidal, indirect_examples,
        min_sim=0.50,  # Lower threshold for indirect (different phrasing by design)
        max_sim=0.95,
    )
    logger.info(f"  Generated: {len(indirect_examples)}, passed filter: {len(passed_indirect)}")

    suicidal_label_id = int(train_df[train_df["label"] == "SUICIDAL_THOUGHTS"]["label_id"].iloc[0])
    for para, sim in passed_indirect:
        all_augmented.append({
            "clean_text": para,
            "label": "SUICIDAL_THOUGHTS",
            "label_id": suicidal_label_id,
            "source": "augmented_indirect",
            "similarity_score": sim,
        })

    # ── Save augmented data ──
    aug_df = pd.DataFrame(all_augmented)

    # Add required columns for compatibility with training pipeline
    aug_df["post_id"] = [f"aug_{i}" for i in range(len(aug_df))]
    aug_df["sentence_id"] = [f"aug_s_{i}" for i in range(len(aug_df))]
    aug_df["sentence_text"] = aug_df["clean_text"]

    aug_df.to_csv(output_dir / "augmented_samples.csv", index=False)

    # Also create the combined training set (original + augmented)
    orig_df = train_df[["post_id", "sentence_id", "sentence_text", "clean_text", "label", "label_id"]].copy()
    orig_df["source"] = "original"
    orig_df["similarity_score"] = 1.0

    combined = pd.concat([orig_df, aug_df[orig_df.columns]], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(output_dir / "train_augmented.csv", index=False)

    # Save metadata
    meta = {
        "model": args.model,
        "original_samples": len(train_df),
        "augmented_samples": len(aug_df),
        "combined_samples": len(combined),
        "paraphrases_per_sentence": args.paraphrases_per_sentence,
        "similarity_range": [args.min_similarity, args.max_similarity],
        "per_class": {
            cls: {
                "original": int(class_counts.get(cls, 0)),
                "augmented": int(len(aug_df[aug_df["label"] == cls])),
                "combined": int(len(combined[combined["label"] == cls])),
            }
            for cls in set(list(target_classes.keys()) + ["SUICIDAL_THOUGHTS"])
        },
    }

    with open(output_dir / "augmentation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original training samples: {len(train_df)}")
    print(f"Augmented samples added:   {len(aug_df)}")
    print(f"Combined training set:     {len(combined)}")
    print("\nPer-class breakdown:")
    for cls in sorted(meta["per_class"].keys()):
        info = meta["per_class"][cls]
        print(f"  {cls}: {info['original']} → {info['combined']} (+{info['augmented']})")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
