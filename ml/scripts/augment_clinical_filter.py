"""
High-quality data augmentation using clinical validity filtering.

Instead of similarity-based filtering (which rejects diverse but valid sentences),
uses an LLM clinical expert to verify each generated sentence actually expresses
the target DSM-5 symptom.

Strategy:
1. Generate sentences from DSM-5 definitions (diverse, not paraphrases)
2. Filter by clinical validity: "Does this express SYMPTOM_X? YES/NO + confidence"
3. Keep only clinically valid sentences (confidence > 0.8)

This produces diverse, high-quality training data that passes clinical scrutiny
even when the vocabulary differs from existing training sentences.

Usage:
    python augment_clinical_filter.py --count 80
"""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMPTOM_CONFIGS = {
    "PSYCHOMOTOR": {
        "label_id": 4,
        "definition": (
            "Observable slowing of physical movement, speech, or thought (retardation), "
            "OR physical restlessness and agitation (pacing, hand-wringing, inability to sit still). "
            "Must be observable by others, not just a subjective feeling."
        ),
        "generate_prompt": (
            "Generate {count} unique Reddit-style first-person sentences where someone describes "
            "psychomotor changes. Include BOTH types:\n"
            "- RETARDATION: moving slowly, speaking slowly, body feels heavy/like lead, "
            "taking forever to do simple things, others noticing you're slow\n"
            "- AGITATION: can't sit still, pacing, fidgeting, nervous energy, restlessness, "
            "hand-wringing, feeling wired\n\n"
            "IMPORTANT: Focus on PHYSICAL/OBSERVABLE behaviors, not just feeling tired (that's fatigue).\n"
            "Vary: intensity, vocabulary, directness, which subtype.\n"
            "Each must be a single sentence. Return ONLY a JSON array of strings."
        ),
        "validate_prompt": (
            "Does this sentence describe psychomotor retardation or psychomotor agitation "
            "as defined by DSM-5 for Major Depressive Disorder?\n\n"
            "VALID examples (these ARE psychomotor):\n"
            "- 'I feel like I'm moving through mud' → YES (slowed physical movement)\n"
            "- 'My friends say I talk really slowly now' → YES (observable speech retardation)\n"
            "- 'I can't stop pacing around the house' → YES (psychomotor agitation)\n"
            "- 'It takes me 10 minutes to get out of a chair' → YES (retardation)\n\n"
            "INVALID examples (these are NOT psychomotor):\n"
            "- 'I feel so tired all the time' → NO (this is FATIGUE)\n"
            "- 'I'm so sad and empty' → NO (this is DEPRESSED MOOD)\n"
            "- 'I can't sleep at night' → NO (this is SLEEP ISSUES)\n"
            "- 'I have no energy' → NO (this is FATIGUE, not observable movement change)\n\n"
            "KEY DISTINCTION: Psychomotor = OBSERVABLE physical movement/speech changes. "
            "Fatigue = feeling tired/no energy. They are different DSM-5 criteria.\n\n"
            'Sentence: "{sentence}"\n\n'
            'Return JSON only: {{"valid": true/false, "confidence": 0.0-1.0, "reason": "brief"}}'
        ),
    },
    "COGNITIVE_ISSUES": {
        "label_id": 7,
        "definition": (
            "Diminished ability to think, concentrate, or make decisions. "
            "Brain fog, difficulty focusing, indecisiveness, memory problems. "
            "Unable to complete tasks, losing track of conversations."
        ),
        "generate_prompt": (
            "Generate {count} unique Reddit-style first-person sentences where someone describes "
            "cognitive difficulties related to depression. Include:\n"
            "- Concentration problems: can't focus, mind wandering, brain fog\n"
            "- Decision-making difficulty: indecisive, paralyzed by choices\n"
            "- Memory issues: forgetting things, losing track of conversations\n"
            "- Mental slowness: feeling mentally dull, can't process information\n\n"
            "IMPORTANT: Focus on COGNITIVE symptoms, not fatigue or sadness.\n"
            "Each must be a single sentence. Return ONLY a JSON array of strings."
        ),
        "validate_prompt": (
            "Does this sentence describe cognitive difficulties as defined by DSM-5 "
            "for Major Depressive Disorder?\n\n"
            "VALID examples (these ARE cognitive issues):\n"
            "- 'I can't focus on anything for more than 5 minutes' → YES (concentration)\n"
            "- 'I keep forgetting what I was saying mid-sentence' → YES (memory)\n"
            "- 'I can't make even simple decisions anymore' → YES (indecisiveness)\n"
            "- 'My brain feels like it's in a fog' → YES (brain fog)\n"
            "- 'I read the same paragraph three times and nothing sticks' → YES (processing)\n\n"
            "INVALID examples (these are NOT cognitive issues):\n"
            "- 'I feel so tired all the time' → NO (this is FATIGUE)\n"
            "- 'I don't enjoy anything anymore' → NO (this is ANHEDONIA)\n"
            "- 'I feel worthless' → NO (this is WORTHLESSNESS)\n"
            "- 'I don't know what to do with my life' → NO (this is general distress, not cognitive)\n\n"
            "KEY DISTINCTION: Cognitive = difficulty with THINKING, CONCENTRATING, DECIDING, REMEMBERING. "
            "Not emotional distress, not fatigue, not loss of interest.\n\n"
            'Sentence: "{sentence}"\n\n'
            'Return JSON only: {{"valid": true/false, "confidence": 0.0-1.0, "reason": "brief"}}'
        ),
    },
}


async def generate_sentences(client, symptom: str, count: int, model: str) -> list[str]:
    """Generate original sentences from DSM-5 definition."""
    config = SYMPTOM_CONFIGS[symptom]
    all_sentences = []
    batch_size = 20

    for _ in range(0, count, batch_size):
        remaining = min(batch_size, count - len(all_sentences))
        prompt = config["generate_prompt"].format(count=remaining)

        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in mental health text. Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
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
                        sentences = json.loads(match.group())
                        all_sentences.extend([s for s in sentences if isinstance(s, str) and len(s) > 20])
                        break
            except Exception as e:
                logger.warning(f"  Gen attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2)

        await asyncio.sleep(0.5)

    return list(set(all_sentences))[:count]


async def validate_sentence(client, sentence: str, symptom: str, model: str) -> dict | None:
    """Validate a single sentence using LLM clinical expert."""
    config = SYMPTOM_CONFIGS[symptom]
    prompt = config["validate_prompt"].format(sentence=sentence)

    for _attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a licensed clinical psychologist. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if content:
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception:
            await asyncio.sleep(1)

    return None


async def main():
    parser = argparse.ArgumentParser(description="High-quality augmentation with clinical filtering")
    parser.add_argument("--gen-model", type=str, default="gemini-3-flash-preview",
                        help="Model for generation (needs creativity)")
    parser.add_argument("--val-model", type=str, default="gemini-3-flash-preview",
                        help="Model for validation (clinical reasoning)")
    parser.add_argument("--count", type=int, default=80, help="Sentences to generate per class")
    parser.add_argument("--min-confidence", type=float, default=0.8,
                        help="Minimum clinical validity confidence (default: 0.8)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "cleaned_v2"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "redsm5" / "augmented_v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")

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

    for symptom, config in SYMPTOM_CONFIGS.items():
        current_count = len(train[train["label"] == symptom])
        logger.info(f"\n{'='*60}")
        logger.info(f"{symptom}: {current_count} existing samples")
        logger.info(f"Generating {args.count} candidates with {args.gen_model}")

        # Step 1: Generate
        candidates = await generate_sentences(client, symptom, args.count, args.gen_model)
        logger.info(f"  Generated: {len(candidates)} unique candidates")

        # Deduplicate against existing training data
        existing_texts = set(train[train["label"] == symptom]["clean_text"].str.lower())
        candidates = [c for c in candidates if c.lower() not in existing_texts]
        logger.info(f"  After dedup vs training: {len(candidates)}")

        # Step 2: Validate with Pro
        logger.info(f"  Validating with {args.val_model} (clinical filter, min_conf={args.min_confidence})...")
        semaphore = asyncio.Semaphore(args.concurrency)
        validated = []
        done = 0

        async def validate_one(sentence: str, *, _semaphore=semaphore, _symptom=symptom, _candidates=candidates):
            nonlocal done
            async with _semaphore:
                result = await validate_sentence(client, sentence, _symptom, args.val_model)
                await asyncio.sleep(0.5)
                done += 1
                pct = done / len(_candidates) * 100
                bar_len = 30
                filled = int(bar_len * done / len(_candidates))
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {done}/{len(_candidates)} ({pct:.0f}%)", end="", flush=True)
                return sentence, result

        tasks = [validate_one(c) for c in candidates]
        results = await asyncio.gather(*tasks)
        print()

        for sentence, validation in results:
            if validation and validation.get("valid") and float(validation.get("confidence", 0)) >= args.min_confidence:
                validated.append({
                    "sentence": sentence,
                    "confidence": float(validation["confidence"]),
                    "reason": validation.get("reason", ""),
                })

        logger.info(f"  Clinically validated: {len(validated)}/{len(candidates)} ({len(validated)/max(len(candidates),1)*100:.0f}%)")

        # Show some examples
        for v in validated[:3]:
            logger.info(f"    ✓ conf={v['confidence']:.2f} \"{v['sentence'][:70]}...\"")

        # Add to results
        label_id = config["label_id"]
        for v in validated:
            all_new.append({
                "post_id": f"clin_{symptom.lower()}_{len(all_new)}",
                "sentence_id": f"clin_s_{symptom.lower()}_{len(all_new)}",
                "sentence_text": v["sentence"],
                "clean_text": v["sentence"],
                "label": symptom,
                "label_id": label_id,
                "source": "clinical_validated",
                "similarity_score": v["confidence"],
            })

    new_df = pd.DataFrame(all_new)

    # Merge with existing augmented data
    existing_aug_path = output_dir / "augmented_samples.csv"
    if existing_aug_path.exists():
        existing_aug = pd.read_csv(existing_aug_path)
        combined_aug = pd.concat([existing_aug, new_df], ignore_index=True)
        # Deduplicate across old + new
        combined_aug = combined_aug.drop_duplicates(subset=["clean_text"], keep="first")
        combined_aug.to_csv(output_dir / "augmented_samples_final.csv", index=False)
        logger.info(f"\nMerged: {len(existing_aug)} existing + {len(new_df)} new = {len(combined_aug)} total")
    else:
        new_df.to_csv(output_dir / "augmented_samples_final.csv", index=False)
        combined_aug = new_df

    # Report
    print(f"\n{'='*60}")
    print("CLINICAL-FILTERED AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"New clinically validated samples: {len(new_df)}")
    for symptom in SYMPTOM_CONFIGS:
        count = len(new_df[new_df["label"] == symptom])
        existing = len(train[train["label"] == symptom])
        total_aug = len(combined_aug[combined_aug["label"] == symptom])
        print(f"  {symptom}: {existing} original + {total_aug} augmented = {existing + total_aug} total")
    print(f"\nSaved to: {output_dir / 'augmented_samples_final.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
