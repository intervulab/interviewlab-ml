#!/usr/bin/env python3
"""
Fine-tune a rubric classifier (DistilBERT) on labeled Q&A pairs.

Usage:
  pip install -r requirements-ml.txt
  python scripts/train_rubric.py --data data/rubric_samples.jsonl --output models/checkpoints/rubric

Expected JSONL rows:
  {"question": "...", "answer": "...", "label": 0-4}
Label maps to score bucket for multi-class rubric grading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT rubric classifier")
    parser.add_argument("--data", required=True, help="Path to JSONL training file")
    parser.add_argument("--output", default="models/checkpoints/rubric")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install ML extras first: pip install -r requirements-ml.txt"
        ) from exc

    rows = []
    with Path(args.data).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit("Training file is empty")

    texts = [f"question: {r['question']} answer: {r['answer']}" for r in rows]
    labels = [int(r["label"]) for r in rows]
    ds = Dataset.from_dict({"text": texts, "label": labels})

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = max(labels) + 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = ds.map(tokenize, batched=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Saved rubric model to {out}")


if __name__ == "__main__":
    main()
