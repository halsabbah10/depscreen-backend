"""
Domain-Adaptive Pre-Training (DAPT) for DistilBERT.

Continues masked language model (MLM) pre-training on unlabeled mental health
Reddit text, adapting the encoder to the domain before fine-tuning.

Reference: Gururangan et al. (ACL 2020) — "Don't Stop Pretraining"

Usage:
    python train_dapt.py [options]

Options:
    --model-name: Base model (default: distilbert-base-uncased)
    --corpus: Path to DAPT corpus text file
    --epochs: Number of MLM epochs (default: 3)
    --batch-size: Batch size (default: 32)
    --lr: Learning rate (default: 5e-5)
    --max-length: Max token length (default: 128)
    --output-dir: Where to save DAPT'd model
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextDataset(Dataset):
    """Simple dataset that tokenizes text lines for MLM."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        self.encodings = []
        for text in tqdm(texts, desc="Tokenizing"):
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def load_corpus(corpus_path: Path, min_length: int = 20) -> list[str]:
    """Load corpus from text file. Split on double-newlines (one post per chunk)."""
    with open(corpus_path) as f:
        raw = f.read()

    # Split on double newlines (each post is separated by \n\n)
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) >= min_length]
    logger.info(f"Loaded {len(chunks)} text chunks from {corpus_path}")
    return chunks


def evaluate_perplexity(model, dataloader, device) -> float:
    """Compute perplexity on a dataset (lower = better domain fit)."""
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pre-Training (DAPT)")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--corpus", type=str, default=None, help="Path to corpus .txt file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-split", type=float, default=0.05, help="Fraction for eval (default: 5%)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    corpus_path = Path(args.corpus) if args.corpus else base_dir / "data" / "dapt_corpus" / "dapt_corpus.txt"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models" / "v2_dapt_base"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}...")
    texts = load_corpus(corpus_path)
    if not texts:
        logger.error("No texts loaded. Run collect_dapt_corpus.py first.")
        return

    # Split into train/eval
    n_eval = max(int(len(texts) * args.eval_split), 100)
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]
    logger.info(f"Train: {len(train_texts)} chunks, Eval: {len(eval_texts)} chunks")

    # Tokenizer + Model (MLM head)
    logger.info(f"Loading {args.model_name} for masked language modeling...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)

    # Datasets
    logger.info("Building datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, args.max_length)

    # MLM data collator — handles masking 15% of tokens
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        collate_fn=data_collator, num_workers=num_workers,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # Measure perplexity BEFORE DAPT
    logger.info("Measuring pre-DAPT perplexity on domain text...")
    pre_dapt_ppl = evaluate_perplexity(model, eval_loader, device)
    logger.info(f"Pre-DAPT perplexity: {pre_dapt_ppl:.2f}")

    # Training loop
    logger.info(f"\nStarting DAPT: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    best_ppl = pre_dapt_ppl

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"DAPT Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches
        eval_ppl = evaluate_perplexity(model, eval_loader, device)

        logger.info(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
            f"eval_perplexity={eval_ppl:.2f} "
            f"(pre-DAPT was {pre_dapt_ppl:.2f})"
        )

        if eval_ppl < best_ppl:
            best_ppl = eval_ppl
            # Save best model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  Saved best DAPT model (ppl={best_ppl:.2f})")

    # Final save if never saved (all epochs worse than pre-DAPT, unlikely)
    if not (output_dir / "config.json").exists():
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Save DAPT metadata
    import json
    dapt_meta = {
        "base_model": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "mlm_probability": args.mlm_probability,
        "train_chunks": len(train_texts),
        "eval_chunks": len(eval_texts),
        "pre_dapt_perplexity": pre_dapt_ppl,
        "post_dapt_perplexity": best_ppl,
        "perplexity_reduction": f"{((pre_dapt_ppl - best_ppl) / pre_dapt_ppl) * 100:.1f}%",
    }
    with open(output_dir / "dapt_metadata.json", "w") as f:
        json.dump(dapt_meta, f, indent=2)

    print(f"\n{'='*60}")
    print("DAPT COMPLETE")
    print(f"{'='*60}")
    print(f"Pre-DAPT perplexity:  {pre_dapt_ppl:.2f}")
    print(f"Post-DAPT perplexity: {best_ppl:.2f}")
    print(f"Reduction:            {((pre_dapt_ppl - best_ppl) / pre_dapt_ppl) * 100:.1f}%")
    print(f"\nDAPT model saved to: {output_dir}")
    print(f"Use this as base model for fine-tuning:")
    print(f"  python train_redsm5_model.py --model-name {output_dir}")


if __name__ == "__main__":
    main()
