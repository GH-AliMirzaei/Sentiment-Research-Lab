import gc
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/project10_imdb/cross_domain_amazon_400.json"
DATASET_ID = "SetFit/amazon_reviews_multi_en"
REVISION = "ec73b66"
SEED = 42
SAMPLES_PER_CLASS = 200

MODELS = {
    "RoBERTa": ROOT / "checkpoints/project03_imdb/roberta_full",
    "DeBERTa": ROOT / "checkpoints/project05_imdb/deberta_full",
}


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted, average="macro"
        )),
        "errors": int(np.sum(actual != predicted)),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }


def infer(checkpoint, texts):
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True
    ).to("cuda").eval()

    probabilities = []

    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            inputs = tokenizer(
                batch, truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            ).to("cuda")

            logits = model(**inputs).logits
            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, 1]
                .cpu().float().tolist()
            )

            if (start + len(batch)) % 100 == 0:
                print("Processed:", start + len(batch))

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.asarray(probabilities)


def mcnemar(actual, first, second):
    a = first == actual
    b = second == actual

    first_only = int(np.sum(a & ~b))
    second_only = int(np.sum(~a & b))
    total = first_only + second_only

    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "exact_p_value": float(
            binomtest(first_only, total).pvalue
        ) if total else 1.0,
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Results already exist: {OUT}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    print("Downloading Amazon test split only...")

    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="test.jsonl",
        repo_type="dataset",
        revision=REVISION,
    )

    with open(path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    # Dataset labels: 0..4 correspond to 1..5 stars.
    negative = [r for r in rows if int(r["label"]) in (0, 1)]
    positive = [r for r in rows if int(r["label"]) in (3, 4)]

    if min(len(negative), len(positive)) < SAMPLES_PER_CLASS:
        raise ValueError("Insufficient samples for balanced evaluation.")

    rng = np.random.default_rng(SEED)

    chosen_negative = rng.choice(
        len(negative), SAMPLES_PER_CLASS, replace=False
    )
    chosen_positive = rng.choice(
        len(positive), SAMPLES_PER_CLASS, replace=False
    )

    selected = (
        [(negative[int(i)]["text"], 0) for i in chosen_negative]
        + [(positive[int(i)]["text"], 1) for i in chosen_positive]
    )

    rng.shuffle(selected)

    texts = [str(text) for text, _ in selected]
    actual = np.asarray([label for _, label in selected])

    print("Dataset:", DATASET_ID)
    print("Samples:", len(actual))
    print("GPU:", torch.cuda.get_device_name(0))

    probabilities = {}

    for name, checkpoint in MODELS.items():
        print("\nRunning:", name)
        probabilities[name] = infer(checkpoint, texts)

    validation = json.loads((
        ROOT / "results/project07_imdb/validation_metrics.json"
    ).read_text(encoding="utf-8"))

    weight = float(validation["selected_weight_roberta"])

    if not 0 <= weight <= 1:
        raise ValueError("Invalid ensemble weight.")

    probabilities["Ensemble"] = (
        weight * probabilities["RoBERTa"]
        + (1 - weight) * probabilities["DeBERTa"]
    )

    predictions = {
        name: (prob >= 0.5).astype(int)
        for name, prob in probabilities.items()
    }

    pairs = {
        "Ensemble_vs_RoBERTa": ("Ensemble", "RoBERTa"),
        "Ensemble_vs_DeBERTa": ("Ensemble", "DeBERTa"),
        "RoBERTa_vs_DeBERTa": ("RoBERTa", "DeBERTa"),
    }

    tests = {
        key: mcnemar(actual, predictions[a], predictions[b])
        for key, (a, b) in pairs.items()
    }

    ordered = sorted(tests, key=lambda k: tests[k]["exact_p_value"])
    adjusted = 0.0

    for i, key in enumerate(ordered):
        adjusted = max(
            adjusted,
            min(1.0, (len(ordered) - i) * tests[key]["exact_p_value"])
        )
        tests[key]["holm_adjusted_p_value"] = adjusted

    report = {
        "project": "10",
        "dataset": DATASET_ID,
        "revision": REVISION,
        "split": "test",
        "samples": len(actual),
        "seed": SEED,
        "label_mapping": {
            "0-1": "negative",
            "2": "excluded",
            "3-4": "positive",
        },
        "max_length": 256,
        "ensemble_weight_roberta": weight,
        "metrics": {
            name: metrics(actual, pred)
            for name, pred in predictions.items()
        },
        "paired_tests": tests,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n===== PROJECT 10 CROSS DOMAIN RESULTS =====")

    for name, value in report["metrics"].items():
        print(
            f"{name:10s} "
            f"Accuracy={value['accuracy']:.4f} "
            f"Macro-F1={value['macro_f1']:.4f} "
            f"Errors={value['errors']}"
        )

    print("\n===== MCNEMAR TESTS =====")
    print(json.dumps(tests, indent=4))
    print("\nCross-domain evaluation completed successfully!")


if __name__ == "__main__":
    main()