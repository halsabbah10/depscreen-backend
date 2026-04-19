"""
Knowledge Distillation: Generate soft labels from Gemini 2.5 Pro.

For each training sentence, prompts Gemini Pro to estimate the probability
distribution across all 11 DSM-5 symptom classes. These soft labels are used
alongside hard labels during training (Hinton et al., 2015).

Supports:
- Pilot mode (--pilot N): Run on N samples, output inspection report
- Full mode: Generate soft labels for entire training set

Usage:
    python distill_labels.py --pilot 50    # Pilot: inspect quality first
    python distill_labels.py               # Full: all training samples
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── DSM-5 Class Definitions (grounded in project's knowledge base) ───────────

CLASS_DEFINITIONS = {
    "DEPRESSED_MOOD": (
        "Persistent feeling of sadness, emptiness, hopelessness, or emotional numbness. "
        "May present as tearfulness, feeling 'down' or 'blue', irritability, "
        "or a heavy sadness that does not lift with positive events. "
        "Goes beyond ordinary sadness or having a bad day."
    ),
    "ANHEDONIA": (
        "Markedly diminished interest or pleasure in all or almost all activities. "
        "No longer enjoying hobbies, social activities, or pastimes that were once pleasurable. "
        "Feeling indifferent, going through the motions without satisfaction. "
        "The loss is pervasive, not limited to one area."
    ),
    "APPETITE_CHANGE": (
        "Significant change in appetite or weight (increase or decrease) "
        "that is not due to intentional dieting. May manifest as food losing its appeal, "
        "eating significantly more or less than usual, or noticeable weight change."
    ),
    "SLEEP_ISSUES": (
        "Insomnia (difficulty falling or staying asleep) or hypersomnia (sleeping excessively). "
        "Disrupted sleep patterns, difficulty getting out of bed, "
        "fatigue despite adequate sleep, or reliance on sleep to escape."
    ),
    "PSYCHOMOTOR": (
        "Observable slowing of physical movement, speech, or thought (retardation), "
        "OR physical restlessness and agitation (pacing, hand-wringing, inability to sit still). "
        "Must be observable by others, not just subjective feeling."
    ),
    "FATIGUE": (
        "Persistent fatigue or loss of energy nearly every day. "
        "Feeling exhausted, drained, or having no energy even for small tasks. "
        "Everything feels like it requires enormous effort. "
        "Distinct from sleepiness — it's a lack of physical/mental energy."
    ),
    "WORTHLESSNESS": (
        "Feelings of worthlessness or excessive, inappropriate guilt. "
        "Deep belief of being fundamentally flawed, inadequate, or a burden. "
        "Harsh self-criticism disproportionate to the situation. "
        "May include feeling undeserving of good things or blaming self for things outside their control."
    ),
    "COGNITIVE_ISSUES": (
        "Diminished ability to think, concentrate, or make decisions. "
        "Brain fog, difficulty focusing, indecisiveness, memory problems. "
        "May manifest as inability to complete tasks, losing track of conversations, "
        "or feeling mentally 'slow'."
    ),
    "SUICIDAL_THOUGHTS": (
        "Recurrent thoughts of death, suicidal ideation, or self-harm. "
        "Ranges from passive ('I wish I didn't exist', 'everyone would be better off without me') "
        "to active ('I want to kill myself', planning methods). "
        "Includes indirect expressions of wanting to die or not wanting to live."
    ),
    "SPECIAL_CASE": (
        "Text indicates clinical concern related to depression that does not map cleanly "
        "to the 9 standard DSM-5 criteria above. Examples: general distress, "
        "social withdrawal as a primary complaint, or mixed symptom presentation "
        "that doesn't fit a single category."
    ),
    "NO_SYMPTOM": (
        "The sentence does not express any depression-related symptom. "
        "It may be neutral, positive, factual, or discussing depression "
        "without personally expressing symptoms. "
        "Includes sentences that mention mental health topics but do not indicate personal experience."
    ),
}

SYSTEM_PROMPT = """\
You are a licensed clinical psychologist with expertise in DSM-5 depression screening.

TASK: Given a sentence from a Reddit mental health post, estimate the probability \
distribution across 11 symptom categories. Probabilities must sum to 1.0.

RULES:
1. Base assessment ONLY on what the sentence explicitly states or strongly implies.
2. If a sentence clearly maps to ONE symptom, give it 0.75-0.95. Do not spread probability to related symptoms without evidence.
3. If genuinely ambiguous between 2-3 symptoms, distribute probability among them.
4. If no depression-related content, NO_SYMPTOM should get 0.85+.
5. Consider Reddit informal language, slang, sarcasm, and indirect expressions.
6. Sarcasm/irony: "I feel SO great" in a depression subreddit likely indicates the opposite.

CRITICAL BOUNDARY RULES (common mistakes to avoid):
- Crying/tearfulness alone → DEPRESSED_MOOD (not NO_SYMPTOM). Crying IS a manifestation of depressed mood.
- "I don't enjoy anything anymore" → ANHEDONIA (not DEPRESSED_MOOD). Loss of pleasure is anhedonia specifically.
- "I feel sad AND nothing interests me" → Split between DEPRESSED_MOOD and ANHEDONIA.
- "I'm so tired" → FATIGUE (not SLEEP_ISSUES) unless sleep is explicitly mentioned.
- "I feel tired even after sleeping a lot" → Split between SLEEP_ISSUES and FATIGUE (both evidenced).
- "I can't get out of bed" → Could be FATIGUE or PSYCHOMOTOR — split if unclear.
- "Everyone would be better off without me" → SUICIDAL_THOUGHTS (passive ideation).
- "I guess I'm depressed" / "I feel depressed" → DEPRESSED_MOOD (the person IS expressing their mood state, even if casually).
- "I am clinically depressed" / "I was diagnosed with depression" → SPECIAL_CASE (discussing diagnosis, not expressing current mood).
- "Some days I am so depressed I can't function" → DEPRESSED_MOOD (expressing mood impact), NOT ANHEDONIA unless loss of pleasure is explicitly stated.
- Factual or neutral statements about mental health topics → NO_SYMPTOM.
- Recovery/positive statements ("I've been feeling better") → NO_SYMPTOM unless also expressing current struggle.

SYMPTOM DEFINITIONS:
"""

FEW_SHOT_EXAMPLES = """
EXAMPLES (follow this reasoning pattern):

Example 1: "I used to love painting but now I can't even pick up a brush"
Key evidence: "used to love" + "can't even pick up" = loss of previously enjoyed activity
→ {"DEPRESSED_MOOD": 0.05, "ANHEDONIA": 0.85, "APPETITE_CHANGE": 0.0, "SLEEP_ISSUES": 0.0, "PSYCHOMOTOR": 0.0, "FATIGUE": 0.05, "WORTHLESSNESS": 0.0, "COGNITIVE_ISSUES": 0.0, "SUICIDAL_THOUGHTS": 0.0, "SPECIAL_CASE": 0.0, "NO_SYMPTOM": 0.05}

Example 2: "I slept 14 hours and still feel exhausted"
Key evidence: "slept 14 hours" = hypersomnia, "still feel exhausted" = fatigue despite sleep
→ {"DEPRESSED_MOOD": 0.0, "ANHEDONIA": 0.0, "APPETITE_CHANGE": 0.0, "SLEEP_ISSUES": 0.45, "PSYCHOMOTOR": 0.0, "FATIGUE": 0.50, "WORTHLESSNESS": 0.0, "COGNITIVE_ISSUES": 0.0, "SUICIDAL_THOUGHTS": 0.0, "SPECIAL_CASE": 0.0, "NO_SYMPTOM": 0.05}

Example 3: "My therapist increased my Lexapro dose last week"
Key evidence: factual statement about treatment, no symptom expression
→ {"DEPRESSED_MOOD": 0.0, "ANHEDONIA": 0.0, "APPETITE_CHANGE": 0.0, "SLEEP_ISSUES": 0.0, "PSYCHOMOTOR": 0.0, "FATIGUE": 0.0, "WORTHLESSNESS": 0.0, "COGNITIVE_ISSUES": 0.0, "SUICIDAL_THOUGHTS": 0.0, "SPECIAL_CASE": 0.10, "NO_SYMPTOM": 0.90}

Example 4: "I'm such a burden to everyone around me, they'd be better off without me"
Key evidence: "burden" = worthlessness, "better off without me" = passive suicidal ideation
→ {"DEPRESSED_MOOD": 0.0, "ANHEDONIA": 0.0, "APPETITE_CHANGE": 0.0, "SLEEP_ISSUES": 0.0, "PSYCHOMOTOR": 0.0, "FATIGUE": 0.0, "WORTHLESSNESS": 0.35, "COGNITIVE_ISSUES": 0.0, "SUICIDAL_THOUGHTS": 0.60, "SPECIAL_CASE": 0.0, "NO_SYMPTOM": 0.05}

Example 5: "I cried for three hours straight last night"
Key evidence: prolonged crying = emotional expression of depressed mood
→ {"DEPRESSED_MOOD": 0.90, "ANHEDONIA": 0.0, "APPETITE_CHANGE": 0.0, "SLEEP_ISSUES": 0.0, "PSYCHOMOTOR": 0.0, "FATIGUE": 0.0, "WORTHLESSNESS": 0.0, "COGNITIVE_ISSUES": 0.0, "SUICIDAL_THOUGHTS": 0.0, "SPECIAL_CASE": 0.0, "NO_SYMPTOM": 0.10}
"""


def build_user_prompt(sentence: str) -> str:
    """Build the per-sentence user prompt."""
    return (
        f'Sentence: "{sentence}"\n\n'
        "Return ONLY a JSON object with these 11 keys as floats summing to 1.0: "
        "DEPRESSED_MOOD, ANHEDONIA, APPETITE_CHANGE, SLEEP_ISSUES, PSYCHOMOTOR, "
        "FATIGUE, WORTHLESSNESS, COGNITIVE_ISSUES, SUICIDAL_THOUGHTS, SPECIAL_CASE, NO_SYMPTOM."
    )


def build_system_prompt() -> str:
    """Build the full system prompt with DSM-5 definitions + few-shot examples."""
    definitions = "\n".join(f"- **{name}**: {desc}" for name, desc in CLASS_DEFINITIONS.items())
    return SYSTEM_PROMPT + definitions + "\n" + FEW_SHOT_EXAMPLES


async def get_soft_label(
    client: AsyncOpenAI,
    sentence: str,
    model: str = "gemini-3.1-pro-preview",
    max_retries: int = 3,
) -> dict | None:
    """Get soft label distribution from Gemini Pro for one sentence."""
    expected_keys = set(CLASS_DEFINITIONS.keys())

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(sentence)},
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            }
            # response_format not supported on preview models
            if "preview" not in model:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            if not content:
                logger.warning(f"  Empty response (attempt {attempt + 1})")
                continue

            # Robust JSON extraction — handles markdown fences, trailing text,
            # single quotes, JavaScript-style comments, trailing commas
            import re as _re

            # Strip markdown code fences
            content = _re.sub(r"```json\s*", "", content)
            content = _re.sub(r"```\s*$", "", content)

            # Strip single-line comments (// ...)
            content = _re.sub(r"//[^\n]*", "", content)

            # Try direct parse first
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # Extract JSON object with regex (greedy — finds the largest {...})
                json_match = _re.search(r"\{.*\}", content, _re.DOTALL)
                if json_match:
                    raw = json_match.group()
                    # Fix trailing commas before closing brace
                    raw = _re.sub(r",\s*}", "}", raw)
                    # Fix single quotes → double quotes
                    raw = raw.replace("'", '"')
                    parsed = json.loads(raw)
                else:
                    raise ValueError("No JSON object found in response")

            # Validate keys
            if set(parsed.keys()) != expected_keys:
                missing = expected_keys - set(parsed.keys())
                extra = set(parsed.keys()) - expected_keys
                logger.warning(f"  Key mismatch: missing={missing}, extra={extra} (attempt {attempt + 1})")
                continue

            # Validate values are floats
            probs = {}
            for k, v in parsed.items():
                probs[k] = float(v)

            # Normalize to sum to 1.0 (Pro sometimes gives 0.99 or 1.01)
            total = sum(probs.values())
            if total <= 0:
                logger.warning(f"  All-zero probabilities (attempt {attempt + 1})")
                continue
            probs = {k: v / total for k, v in probs.items()}

            return probs

        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse error: {e} (attempt {attempt + 1})")
        except Exception as e:
            if "429" in str(e):
                wait = 30 * (attempt + 1)
                logger.warning(f"  Rate limited, waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.warning(f"  API error: {e} (attempt {attempt + 1})")
                await asyncio.sleep(2)

    return None


async def process_batch(
    client: AsyncOpenAI,
    sentences: list[str],
    model: str,
    concurrency: int = 5,
    delay: float = 3.0,
    resume_from: dict | None = None,
    sentence_ids: list[str] | None = None,
    checkpoint_path: str | None = None,
) -> list[dict | None]:
    """Process sentences with rate limiting.

    Uses sequential processing with a delay between requests to avoid
    hitting API rate limits. Supports resumption via checkpoint file.

    Args:
        delay: Seconds between requests (default 3.0 for ~20 RPM)
        resume_from: Dict of sentence_id → soft_label to skip already-done samples
        sentence_ids: List of sentence IDs aligned to sentences (for resume/checkpoint)
        checkpoint_path: Path to write incremental results for crash recovery
    """
    results = [None] * len(sentences)
    skipped = 0

    # Resume: pre-fill results for already-completed samples
    if resume_from and sentence_ids:
        for i, sid in enumerate(sentence_ids):
            if sid in resume_from:
                results[i] = resume_from[sid]
                skipped += 1
        if skipped > 0:
            logger.info(f"  Resumed: {skipped}/{len(sentences)} already completed")

    # Process remaining with bounded concurrency
    checkpoint_data = dict(resume_from) if resume_from else {}
    semaphore = asyncio.Semaphore(concurrency)
    done_count = skipped
    total = len(sentences)
    lock = asyncio.Lock()

    async def process_one(i: int, sentence: str):
        nonlocal done_count
        async with semaphore:
            result = await get_soft_label(client, sentence, model)
            results[i] = result

            # Rate limiting between requests
            await asyncio.sleep(delay)

            async with lock:
                done_count += 1
                # Save to checkpoint
                if checkpoint_path and sentence_ids and result is not None:
                    checkpoint_data[sentence_ids[i]] = result
                    if done_count % 50 == 0:
                        with open(checkpoint_path, "w") as f:
                            json.dump(checkpoint_data, f)

                # Progress bar
                pct = done_count / total * 100
                bar_len = 40
                filled = int(bar_len * done_count / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {done_count}/{total} ({pct:.1f}%)", end="", flush=True)

    tasks = []
    for i, sentence in enumerate(sentences):
        if results[i] is not None:
            continue
        tasks.append(process_one(i, sentence))

    if tasks:
        await asyncio.gather(*tasks)
        print()  # newline after progress bar

    # Final checkpoint save
    if checkpoint_path and sentence_ids:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

    return results


def pilot_analysis(
    df: pd.DataFrame,
    soft_labels: list[dict],
    n: int,
) -> dict:
    """Analyze pilot results: agreement, distribution quality, edge cases."""
    from sklearn.metrics import cohen_kappa_score

    valid_indices = [i for i in range(n) if soft_labels[i] is not None]
    success_rate = len(valid_indices) / n

    if not valid_indices:
        return {"success_rate": 0, "error": "No valid soft labels returned"}

    # Compare Pro's argmax vs human label
    human_labels = []
    pro_labels = []
    agreements = []
    disagreements = []
    entropy_values = []

    for i in valid_indices:
        row = df.iloc[i]
        human_label = row["label"]
        probs = soft_labels[i]

        pro_label = max(probs, key=probs.get)
        human_labels.append(human_label)
        pro_labels.append(pro_label)

        # Entropy: low = confident, high = uncertain
        prob_values = np.array(list(probs.values()))
        prob_values = np.clip(prob_values, 1e-10, 1.0)  # avoid log(0)
        entropy = -np.sum(prob_values * np.log2(prob_values))
        entropy_values.append(entropy)

        if pro_label == human_label:
            agreements.append(i)
        else:
            disagreements.append(
                {
                    "index": i,
                    "sentence": row["clean_text"][:100],
                    "human": human_label,
                    "pro_argmax": pro_label,
                    "pro_confidence": probs[pro_label],
                    "human_prob": probs[human_label],
                    "entropy": entropy,
                }
            )

    # Cohen's Kappa
    kappa = cohen_kappa_score(human_labels, pro_labels)

    # Agreement rate
    agreement_rate = len(agreements) / len(valid_indices)

    # Average entropy
    avg_entropy = np.mean(entropy_values)

    # Soft label quality: for agreed samples, what's the average confidence?
    agreed_confidences = []
    for i in agreements:
        probs = soft_labels[i]
        label = df.iloc[i]["label"]
        agreed_confidences.append(probs[label])

    report = {
        "pilot_size": n,
        "valid_responses": len(valid_indices),
        "success_rate": success_rate,
        "agreement_rate": agreement_rate,
        "cohens_kappa": kappa,
        "avg_entropy": avg_entropy,
        "avg_confidence_when_agreed": np.mean(agreed_confidences) if agreed_confidences else 0,
        "num_disagreements": len(disagreements),
        "disagreements": disagreements[:20],  # top 20 for inspection
    }

    return report


def print_pilot_report(report: dict):
    """Print formatted pilot analysis."""
    print("\n" + "=" * 70)
    print("KNOWLEDGE DISTILLATION — PILOT REPORT")
    print("=" * 70)

    print(f"\n  Pilot size:                {report['pilot_size']}")
    print(f"  Valid API responses:       {report['valid_responses']} ({report['success_rate']:.1%})")
    print(f"  Agreement rate (Pro=Human): {report['agreement_rate']:.1%}")
    print(f"  Cohen's Kappa:             {report['cohens_kappa']:.4f}")
    print(f"  Avg entropy:               {report['avg_entropy']:.3f} bits")
    print(f"  Avg confidence (agreed):   {report['avg_confidence_when_agreed']:.3f}")

    # Kappa interpretation
    kappa = report["cohens_kappa"]
    if kappa >= 0.80:
        interp = "EXCELLENT — near-perfect agreement"
    elif kappa >= 0.60:
        interp = "SUBSTANTIAL — good agreement, soft labels are usable"
    elif kappa >= 0.40:
        interp = "MODERATE — some disagreement, proceed with caution"
    elif kappa >= 0.20:
        interp = "FAIR — significant disagreement, soft labels may be noisy"
    else:
        interp = "POOR — soft labels unreliable, skip distillation"
    print(f"  Kappa interpretation:      {interp}")

    if report["disagreements"]:
        print(f"\n  Disagreements ({report['num_disagreements']} total, showing up to 20):")
        print(f"  {'Idx':>4} {'Human':<22} {'Pro':<22} {'Pro Conf':>8} {'H Prob':>6} {'Entropy':>7}  Sentence")
        print("  " + "-" * 110)
        for d in report["disagreements"]:
            print(
                f"  {d['index']:>4} {d['human']:<22} {d['pro_argmax']:<22} "
                f"{d['pro_confidence']:>8.3f} {d['human_prob']:>6.3f} {d['entropy']:>7.2f}  "
                f"{d['sentence'][:50]}..."
            )

    # Go/no-go
    print(f"\n{'=' * 70}")
    if kappa >= 0.60 and report["success_rate"] >= 0.90:
        print("VERDICT: GO — Soft labels are reliable. Proceed to full generation.")
    elif kappa >= 0.40 and report["success_rate"] >= 0.80:
        print("VERDICT: PROCEED WITH CAUTION — Use higher α (more weight on hard labels).")
    else:
        print("VERDICT: NO-GO — Soft labels too noisy. Skip distillation, document finding.")
    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Generate soft labels from Gemini Pro")
    parser.add_argument("--pilot", type=int, default=None, help="Run pilot on N samples")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent API calls (default: 3)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls (default: 0.5)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "redsm5" / "distilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_csv(data_dir / "train.csv")
    logger.info(f"Loaded {len(train_df)} training samples")

    # Client
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        # Try loading from .env
        env_path = base_dir.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("LLM_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key:
        logger.error("LLM_API_KEY not found. Set it in environment or .env file.")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    if args.pilot:
        # ── PILOT MODE ──
        n = min(args.pilot, len(train_df))
        logger.info(f"\n=== PILOT MODE: {n} samples ===")

        # Stratified sample: pick proportionally from each class
        pilot_indices = []
        for _label, group in train_df.groupby("label"):
            k = max(1, int(n * len(group) / len(train_df)))
            pilot_indices.extend(group.sample(n=k, random_state=42).index.tolist())
        pilot_df = train_df.loc[pilot_indices].head(n).reset_index(drop=True)

        logger.info("Pilot class distribution:")
        for label, count in pilot_df["label"].value_counts().sort_index().items():
            logger.info(f"  {label}: {count}")

        n = len(pilot_df)  # actual count after stratified sampling
        sentences = pilot_df["clean_text"].tolist()

        logger.info(f"\nCalling {args.model} for {n} sentences (concurrency={args.concurrency})...")
        start = time.time()
        soft_labels = await process_batch(client, sentences, args.model, args.concurrency)
        elapsed = time.time() - start
        logger.info(f"Completed in {elapsed:.1f}s ({elapsed / n:.1f}s/sample)")

        # Analyze
        report = pilot_analysis(pilot_df, soft_labels, n)
        print_pilot_report(report)

        # Save pilot data
        pilot_output = {
            "config": {
                "model": args.model,
                "pilot_size": n,
                "elapsed_seconds": elapsed,
            },
            "report": {k: v for k, v in report.items() if k != "disagreements"},
            "disagreements": report.get("disagreements", []),
            "soft_labels": [
                {
                    "sentence": pilot_df.iloc[i]["clean_text"],
                    "human_label": pilot_df.iloc[i]["label"],
                    "soft_label": soft_labels[i],
                }
                for i in range(n)
                if soft_labels[i] is not None
            ],
        }

        with open(output_dir / "pilot_report.json", "w") as f:
            json.dump(pilot_output, f, indent=2, default=str)

        logger.info(f"\nPilot report saved to: {output_dir / 'pilot_report.json'}")

    else:
        # ── FULL MODE ──
        logger.info(f"\n=== FULL MODE: {len(train_df)} samples ===")

        # Check if pilot was run
        pilot_path = output_dir / "pilot_report.json"
        if pilot_path.exists():
            with open(pilot_path) as f:
                pilot = json.load(f)
            kappa = pilot["report"].get("cohens_kappa", 0)
            if kappa < 0.40:
                logger.error(
                    f"Pilot Kappa = {kappa:.3f} (below 0.40 threshold). "
                    "Soft labels too noisy. Aborting. Run pilot analysis first."
                )
                return
            logger.info(f"Pilot passed (Kappa={kappa:.3f}). Proceeding with full generation.")
        else:
            logger.warning("No pilot report found. Run --pilot first for quality validation.")
            logger.warning("Proceeding anyway...")

        sentences = train_df["clean_text"].tolist()
        sentence_ids = train_df["sentence_id"].tolist()
        checkpoint_path = str(output_dir / "distill_checkpoint.json")

        # Load checkpoint for resume if exists
        resume_from = None
        if Path(checkpoint_path).exists():
            with open(checkpoint_path) as f:
                resume_from = json.load(f)
            logger.info(f"Found checkpoint with {len(resume_from)} completed samples — resuming")

        logger.info(f"Calling {args.model} for {len(sentences)} sentences (delay={args.delay}s)...")
        start = time.time()
        soft_labels = await process_batch(
            client,
            sentences,
            args.model,
            args.concurrency,
            delay=args.delay,
            resume_from=resume_from,
            sentence_ids=sentence_ids,
            checkpoint_path=checkpoint_path,
        )
        elapsed = time.time() - start

        # Count successes
        valid = sum(1 for s in soft_labels if s is not None)
        logger.info(f"Completed in {elapsed:.1f}s. Valid: {valid}/{len(sentences)} ({valid / len(sentences):.1%})")

        # Save soft labels alongside training data
        soft_label_rows = []
        for i, row in train_df.iterrows():
            entry = {
                "post_id": row["post_id"],
                "sentence_id": row["sentence_id"],
                "clean_text": row["clean_text"],
                "hard_label": row["label"],
                "hard_label_id": int(row["label_id"]),
            }
            if soft_labels[i] is not None:
                for cls, prob in soft_labels[i].items():
                    entry[f"soft_{cls}"] = prob
                entry["soft_label_valid"] = True
            else:
                for cls in CLASS_DEFINITIONS:
                    entry[f"soft_{cls}"] = 0.0
                entry["soft_label_valid"] = False
            soft_label_rows.append(entry)

        soft_df = pd.DataFrame(soft_label_rows)
        soft_df.to_csv(output_dir / "train_distilled.csv", index=False)

        # Save metadata
        meta = {
            "model": args.model,
            "total_samples": len(train_df),
            "valid_soft_labels": valid,
            "invalid_soft_labels": len(train_df) - valid,
            "success_rate": valid / len(train_df),
            "elapsed_seconds": elapsed,
            "avg_seconds_per_sample": elapsed / len(train_df),
        }
        with open(output_dir / "distillation_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"\nDistilled labels saved to: {output_dir / 'train_distilled.csv'}")
        logger.info(f"Metadata saved to: {output_dir / 'distillation_metadata.json'}")


if __name__ == "__main__":
    asyncio.run(main())
