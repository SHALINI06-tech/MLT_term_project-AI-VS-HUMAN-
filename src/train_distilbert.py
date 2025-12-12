# src/train_distilbert.py
import argparse, os
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

def main(dataset_path="data/dataset.csv", out_dir="models/distilbert", epochs=3, batch=8):
    df = pd.read_csv(dataset_path)
    df = df[["text","label"]].dropna().reset_index(drop=True)
    ds = Dataset.from_pandas(df)
    # train/test split
    ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    def tokenize(batch): return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format(type="torch")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_steps=50,
        push_to_hub=False
    )
    def compute_metrics(pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        preds = np.argmax(pred.predictions, axis=1)
        labels = pred.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0)
        }
    trainer = Trainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["test"], compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
