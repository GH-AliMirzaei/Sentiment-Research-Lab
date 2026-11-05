
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm.auto import tqdm

# Run from the Sentiment-Research-Lab root directory
MODEL_PATH = Path("checkpoints/project04_imdb/distilbert_full")
TEST_PATH = "data/processed/test"
RESULTS_DIR = Path("results/project04_imdb/distilbert_full")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# Load trained model - do NOT retrain
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH)
).to(device)
model.eval()

# Official IMDB test dataset
test_data = load_from_disk(TEST_PATH)
print("Test samples:", len(test_data))

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
    )

test_data = test_data.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
)

loader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer),
)

all_labels = []
all_predictions = []

with torch.inference_mode():
    for batch in tqdm(loader, desc="Evaluating DistilBERT"):
        labels = batch.pop("labels")
        inputs = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

metrics = {
    "model": "distilbert/distilbert-base-uncased",
    "dataset": "IMDB",
    "split": "test",
    "samples": len(all_labels),
    "accuracy": float(accuracy_score(all_labels, all_predictions)),
    "precision_macro": float(precision),
    "recall_macro": float(recall),
    "f1_macro": float(f1),
    "confusion_matrix": confusion_matrix(
        all_labels, all_predictions, labels=[0, 1]
    ).tolist(),
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(
    RESULTS_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nDistilBERT TEST RESULTS")
print(json.dumps(metrics, indent=4))

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    )
)

print("\nTest results saved successfully!")
