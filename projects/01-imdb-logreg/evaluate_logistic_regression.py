
import json
from pathlib import Path

import joblib
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load trained model
model_path = Path(
    "checkpoints/project01_imdb/logistic_regression.joblib"
)

model = joblib.load(model_path)

# 2. Load official test dataset
test_data = load_from_disk("data/processed/test")

print("Test samples:", len(test_data))

# 3. Run predictions
print("Evaluating trained model on TEST data...")

predictions = model.predict(test_data["text"])
actual = test_data["label"]

# 4. Calculate metrics
metrics = {
    "model": "TF-IDF + Logistic Regression",
    "dataset": "IMDB",
    "split": "test",
    "samples": len(test_data),
    "accuracy": accuracy_score(actual, predictions),
    "precision_macro": precision_score(
        actual, predictions, average="macro"
    ),
    "recall_macro": recall_score(
        actual, predictions, average="macro"
    ),
    "f1_macro": f1_score(
        actual, predictions, average="macro"
    ),
    "confusion_matrix": confusion_matrix(
        actual, predictions
    ).tolist()
}

# 5. Display results
print("\n=== FINAL TEST RESULTS ===")
print(json.dumps(metrics, indent=4))

print("\nClassification Report:")
print(classification_report(
    actual,
    predictions,
    target_names=["Negative", "Positive"]
))

# 6. Save final test metrics
output_dir = Path("results/project01_imdb/logreg")
output_dir.mkdir(parents=True, exist_ok=True)

with open(
    output_dir / "test_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nFinal test results saved successfully!")
