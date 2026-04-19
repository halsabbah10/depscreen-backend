"""
Generate + validate augmented data using the DL model itself as filter.

Strategy: Generate sentences via LLM, then keep only those our DistilBERT
model classifies as the target class (top-3 predictions, prob > 0.1).

This produces training data the model can learn from — it reinforces
patterns the model already partially recognizes.

Usage:
    python augment_dl_validated.py
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from openai import AsyncOpenAI
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_redsm5_model import SymptomClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TARGETS = {
    "PSYCHOMOTOR": {
        "label_id": 4,
        "count": 100,
        "definition": (
            "Observable slowing of physical movement/speech (retardation) "
            "OR restlessness/agitation (pacing, fidgeting, can't sit still). "
            "Must be PHYSICAL/OBSERVABLE, not just feeling tired."
        ),
    },
    "COGNITIVE_ISSUES": {
        "label_id": 7,
        "count": 100,
        "definition": (
            "Difficulty concentrating, brain fog, indecisiveness, memory problems, "
            "losing track of conversations, feeling mentally slow/dull."
        ),
    },
}


async def generate_sentences(client, symptom, definition, count, model="gemini-3-flash-preview"):
    all_sentences = []
    prompt = (
        f"Generate {count} unique Reddit-style first-person sentences where someone describes "
        f"{symptom.replace('_', ' ').lower()} symptoms.\n\n"
        f"Definition: {definition}\n\n"
        f"Use informal language. Each must be a single sentence. Vary intensity and vocabulary.\n"
        f"Return ONLY a JSON array of strings."
    )
    for _ in range(0, count, 25):
        for _attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4096,
                    temperature=0.9,
                )
                content = r.choices[0].message.content or ""
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```\s*$", "", content)
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    sents = json.loads(match.group())
                    all_sentences.extend([s for s in sents if isinstance(s, str) and len(s) > 20])
                    break
            except Exception:
                await asyncio.sleep(1)
        await asyncio.sleep(0.5)
    return list(set(all_sentences))


def validate_with_model(sentences, target_label_id, model, tokenizer, device, top_k=3, min_prob=0.1):
    """Keep sentences where our DL model puts target class in top-K predictions."""
    passed = []
    model.eval()
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(sent, truncation=True, max_length=128, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)[0]
            top_k_ids = torch.topk(probs, top_k).indices.tolist()
            target_prob = probs[target_label_id].item()

            if target_label_id in top_k_ids and target_prob > min_prob:
                passed.append((sent, target_prob))
    return sorted(passed, key=lambda x: -x[1])


async def main():
    base_dir = Path(__file__).parent.parent
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load model
    print("Loading DL model for validation...")
    with open(base_dir / "models" / "baseline_v1" / "redsm5_metadata.json") as f:
        meta = json.load(f)
    label_map = meta["label_map"]

    dl_model = SymptomClassifier(num_classes=11, model_name=str(base_dir / "models" / "v2_dapt_base"))
    dl_model.load_state_dict(torch.load(base_dir / "models" / "baseline_v1" / "symptom_classifier.pt", map_location=device))
    dl_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir / "models" / "v2_dapt_base"))

    # LLM client
    api_key = ""
    env_path = base_dir.parent / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("LLM_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break
    client = AsyncOpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    all_new = []

    for symptom, config in TARGETS.items():
        print(f"\n{'='*50}")
        print(f"{symptom} (label_id={config['label_id']})")

        # Generate
        print(f"  Generating {config['count']} candidates...")
        candidates = await generate_sentences(client, symptom, config["definition"], config["count"])
        print(f"  Generated: {len(candidates)} unique")

        # DL-model validate
        print("  Validating with DL model (top-3, min_prob=0.1)...")
        passed = validate_with_model(candidates, config["label_id"], dl_model, tokenizer, device)
        print(f"  Passed: {len(passed)}/{len(candidates)} ({len(passed)/max(len(candidates),1)*100:.0f}%)")

        for s, p in passed[:5]:
            print(f"    prob={p:.3f} \"{s[:70]}\"")

        for sent, prob in passed:
            all_new.append({
                "post_id": f"dlval_{symptom.lower()}_{len(all_new)}",
                "sentence_id": f"dlval_s_{symptom.lower()}_{len(all_new)}",
                "sentence_text": sent,
                "clean_text": sent,
                "label": symptom,
                "label_id": config["label_id"],
                "source": "dl_validated",
                "similarity_score": prob,
            })

    new_df = pd.DataFrame(all_new)

    # Merge with existing augmented
    aug_dir = base_dir / "data" / "redsm5" / "augmented_v4"
    for fname in ["augmented_samples_final.csv", "augmented_samples.csv"]:
        aug_path = aug_dir / fname
        if aug_path.exists():
            existing = pd.read_csv(aug_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["clean_text"], keep="first")
            combined.to_csv(aug_dir / "augmented_samples_final.csv", index=False)
            break
    else:
        combined = new_df
        combined.to_csv(aug_dir / "augmented_samples_final.csv", index=False)

    print(f"\n{'='*50}")
    print("COMPLETE")
    print(f"New DL-validated: {len(new_df)}")
    for sym in TARGETS:
        count = len(combined[combined["label"] == sym])
        print(f"  {sym}: {count} total augmented")
    print(f"Total augmented: {len(combined)}")
    print(f"Saved to: {aug_dir / 'augmented_samples_final.csv'}")

    del dl_model
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    asyncio.run(main())
