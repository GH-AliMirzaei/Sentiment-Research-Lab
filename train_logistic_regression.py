
import json
from pathlib import Path

import joblib
from datasets import load_from_disk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load prepared IMDB data
train = load_from_disk("data/processed/train")
validation = load_from_disk("data/processed/validation")

print("Training samples:", len(train))
print("Validation samples:", len(validation))

# 2. Create text classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 3. Train model
print("Training Logistic Regression...")
model.fit(train["text"], train["label"])

# 4. Validate model
print("Evaluating on validation data...")
predictions = model.predict(validation["text"])
actual = validation["label"]

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "dataset": "IMDB",
    "split": "validation",
    "accuracy": accuracy_score(actual, predictions),
    "precision": precision_score(actual, predictions),
    "recall": recall_score(actual, predictions),
    "f1_score": f1_score(actual, predictions),
    "confusion_matrix": confusion_matrix(
        actual, predictions
    ).tolist()
}

# 5. Display results
print("\n=== RESULTS ===")
print(json.dumps(metrics, indent=4))

print("\nClassification Report:")
print(classification_report(
    actual,
    predictions,
    target_names=["Negative", "Positive"]
))

# 6. Save results and trained model
results_dir = Path("results/project01_imdb/logreg")
model_dir = Path("checkpoints/project01_imdb")

results_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

joblib.dump(
    model,
    model_dir / "logistic_regression.joblib"
)

print("\nModel and results saved successfully!")
