import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Configuration
MODEL_NAME = "distilbert/distilbert-base-uncased"
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print("GPU:", torch.cuda.get_device_name(0))

# Load previously prepared IMDB dataset
train_data = load_from_disk("data/processed/train")
val_data = load_from_disk("data/processed/validation")

if args.smoke_test:
    train_data = train_data.shuffle(seed=42).select(range(512))
    val_data = val_data.shuffle(seed=42).select(range(200))

print("Train:", len(train_data))
print("Validation:", len(val_data))

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
    )

train_data = train_data.map(
    tokenize, batched=True, remove_columns=["text"]
)
val_data = val_data.map(
    tokenize, batched=True, remove_columns=["text"]
)

# Load pretrained DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

run_name = "distilbert_smoke" if args.smoke_test else "distilbert_full"

output_dir = f"checkpoints/project04_imdb/{run_name}"
results_dir = Path(f"results/project04_imdb/{run_name}")
results_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1 if args.smoke_test else 2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    ),
    compute_metrics=compute_metrics,
)

print("Starting DistilBERT training...")
trainer.train()

metrics = trainer.evaluate()

with open(
    results_dir / "validation_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training completed!")
print(json.dumps(metrics, indent=4))
