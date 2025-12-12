#!/usr/bin/env python3
"""
train.py

Fine-tune DistilBERT on dataset.csv.
Usage:
    python train.py --dataset dataset.csv --epochs 3 --output_dir model
"""

import argparse
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def main(dataset_path, epochs, output_dir):
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    import torch
    from sklearn.model_selection import train_test_split
    from datasets import Dataset

    print("Loading data:", dataset_path)
    df = pd.read_csv(dataset_path)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns(["text", "__index_level_0__"], errors="ignore")
    val_ds = val_ds.remove_columns(["text", "__index_level_0__"], errors="ignore")

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Robust TrainingArguments creation: try both arg names
    args_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Try the safer 'evaluation_strategy' first, fall back to 'eval_strategy' if needed
    try:
        print("Creating TrainingArguments with 'evaluation_strategy'...")
        training_args = TrainingArguments(**{**args_kwargs, "evaluation_strategy": "epoch"})
    except TypeError:
        print("Falling back to 'eval_strategy' key name...")
        training_args = TrainingArguments(**{**args_kwargs, "eval_strategy": "epoch"})

    def compute_metrics(p):
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model to", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete. Model saved to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="model")
    args = parser.parse_args()
    main(args.dataset, args.epochs, args.output_dir)
