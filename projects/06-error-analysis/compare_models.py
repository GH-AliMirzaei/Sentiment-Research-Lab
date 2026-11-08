import csv
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data/processed/test"
RESULT_DIR = ROOT / "results/project06_imdb/paired_roberta_deberta"

MODELS = {
    "RoBERTa": ROOT / "checkpoints/project03_imdb/roberta_full",
    "DeBERTa": ROOT / "checkpoints/project05_imdb/deberta_full",
}

BATCH_SIZE = 8
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

for name, path in MODELS.items():
    if not path.is_dir():
        raise FileNotFoundError(f"{name} checkpoint not found: {path}")

print("Device:", DEVICE)
test = load_from_disk(str(TEST_PATH))
labels = [int(x) for x in test["label"]]
print("Test samples:", len(labels))


def predict(model_name, checkpoint):
    print(f"\nEvaluating {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint)
    ).to(DEVICE)
    model.eval()

    predictions = []
    start_time = time.perf_counter()

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(test), BATCH_SIZE),
            desc=model_name
        ):
            texts = test[start:start + BATCH_SIZE]["text"]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(DEVICE)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)
            predictions.extend(
                outputs.logits.argmax(dim=-1).cpu().tolist()
            )

    elapsed = time.perf_counter() - start_time

    del model, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, elapsed


roberta_pred, roberta_time = predict(
    "RoBERTa", MODELS["RoBERTa"]
)

deberta_pred, deberta_time = predict(
    "DeBERTa", MODELS["DeBERTa"]
)

if not (len(labels) == len(roberta_pred) == len(deberta_pred)):
    raise ValueError("Prediction and label counts do not match.")

both_correct = 0
both_wrong = 0
roberta_only_correct = 0
deberta_only_correct = 0

errors_path = RESULT_DIR / "paired_errors.csv"

with errors_path.open(
    "w", newline="", encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)
    writer.writerow([
        "index",
        "actual",
        "roberta_prediction",
        "deberta_prediction",
        "roberta_correct",
        "deberta_correct",
        "review_excerpt"
    ])

    for i, (actual, rb, db) in enumerate(
        zip(labels, roberta_pred, deberta_pred)
    ):
        rc = rb == actual
        dc = db == actual

        if rc and dc:
            both_correct += 1
        elif not rc and not dc:
            both_wrong += 1
        elif rc and not dc:
            roberta_only_correct += 1
        else:
            deberta_only_correct += 1

        if not (rc and dc):
            writer.writerow([
                i,
                actual,
                rb,
                db,
                rc,
                dc,
                test[i]["text"][:600].replace("\n", " ")
            ])


def model_metrics(predictions, runtime):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(
            labels, predictions, average="macro"
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "inference_seconds": runtime
    }


discordant = roberta_only_correct + deberta_only_correct

if discordant > 0:
    p_value = binomtest(
        roberta_only_correct,
        n=discordant,
        p=0.5,
        alternative="two-sided"
    ).pvalue
else:
    p_value = 1.0

results = {
    "dataset": "IMDB",
    "split": "test",
    "samples": len(labels),
    "max_length": MAX_LENGTH,
    "roberta": model_metrics(roberta_pred, roberta_time),
    "deberta": model_metrics(deberta_pred, deberta_time),
    "paired_comparison": {
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "roberta_only_correct": roberta_only_correct,
        "deberta_only_correct": deberta_only_correct,
        "discordant_samples": discordant,
        "mcnemar_exact_p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05)
    }
}

metrics_path = RESULT_DIR / "metrics.json"

with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n===== FINAL STATISTICAL COMPARISON =====")
print(json.dumps(results, indent=4))
print("\nSaved:", metrics_path)
print("Saved:", errors_path)
print("Analysis completed successfully!")