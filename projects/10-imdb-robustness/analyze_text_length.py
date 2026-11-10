import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/project07_imdb/cache"
OUT = ROOT / "results/project10_imdb"
BERT = ROOT / "checkpoints/project03_imdb/roberta_full"


def read_cache(name, labels):
    path = CACHE / f"validation_{name}.npz"

    with np.load(path, allow_pickle=False) as data:
        cached_labels = data["labels"].copy()
        probabilities = data["probabilities"].copy()
        max_length = int(data["max_length"])

    if not np.array_equal(cached_labels, labels):
        raise ValueError(f"Label mismatch for {name}")

    if max_length != 256:
        raise ValueError("Unexpected model max_length")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite probabilities")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Invalid probabilities")

    return probabilities


def calculate_metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual,
            predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )),
        "correct": int(np.sum(actual == predicted)),
        "errors": int(np.sum(actual != predicted)),
    }


def main():
    output = OUT / "length_analysis_validation.json"

    if output.exists():
        raise FileExistsError(
            f"Existing report preserved: {output}"
        )

    dataset = load_from_disk(
        str(ROOT / "data/processed/validation")
    )

    labels = np.asarray(dataset["label"], dtype=np.int64)

    roberta_prob = read_cache("roberta", labels)
    deberta_prob = read_cache("deberta", labels)

    config_path = (
        ROOT / "results/project07_imdb/validation_metrics.json"
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    weight = float(config["selected_weight_roberta"])

    if not 0 <= weight <= 1:
        raise ValueError("Invalid ensemble weight")

    predictions = {
        "RoBERTa": (roberta_prob >= 0.5).astype(int),
        "DeBERTa": (deberta_prob >= 0.5).astype(int),
        "Ensemble": (
            weight * roberta_prob
            + (1 - weight) * deberta_prob >= 0.5
        ).astype(int),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT),
        local_files_only=True,
        use_fast=True,
    )

    texts = dataset["text"]
    lengths = []

    for start in range(0, len(texts), 128):
        batch = texts[start:start + 128]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        lengths.extend(encoded["length"])

    lengths = np.asarray(lengths, dtype=np.int64)

    if len(lengths) != len(labels):
        raise ValueError("Token length count mismatch")

    groups = {
        "Short (<=128)": lengths <= 128,
        "Medium (129-256)": (
            (lengths > 128) & (lengths <= 256)
        ),
        "Long (>256)": lengths > 256,
    }

    report = {
        "project": "10",
        "dataset": "IMDB",
        "split": "validation",
        "samples": int(len(labels)),
        "length_reference": (
            "RoBERTa tokenizer, including special tokens"
        ),
        "model_max_length": 256,
        "ensemble_weight_roberta": weight,
        "groups": {},
    }

    print("\n===== PROJECT 10 LENGTH ROBUSTNESS =====")
    print("Total samples:", len(labels))
    print()

    for name, mask in groups.items():
        count = int(mask.sum())

        if count == 0:
            raise ValueError(f"Empty length group: {name}")

        results = {
            model: calculate_metrics(
                labels[mask],
                pred[mask],
            )
            for model, pred in predictions.items()
        }

        report["groups"][name] = {
            "samples": count,
            "positive": int(np.sum(labels[mask] == 1)),
            "negative": int(np.sum(labels[mask] == 0)),
            "metrics": results,
            "ensemble_vs_roberta_delta_pp": round(
                100 * (
                    results["Ensemble"]["accuracy"]
                    - results["RoBERTa"]["accuracy"]
                ),
                4,
            ),
        }

        print(f"{name} | samples={count}")

        for model, result in results.items():
            print(
                f"  {model:10s} "
                f"Accuracy={result['accuracy']:.4f} "
                f"Macro-F1={result['macro_f1']:.4f} "
                f"Errors={result['errors']}"
            )

        print()

    OUT.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("Saved:", output)
    print("Project 10 length analysis completed successfully!")


if __name__ == "__main__":
    main()