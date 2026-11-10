import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/project08_imdb"
CACHE = ROOT / "results/project07_imdb/cache"
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--samples", type=int, default=200)
SAMPLES = PARSER.parse_args().samples

if SAMPLES < 2 or SAMPLES % 2 != 0 or SAMPLES > 5000:
    raise ValueError("Samples must be even and between 2 and 5000.")
SEED = 42


def read_llm(mode, indices, labels):
    path = OUT / f"qwen15b_{mode}_predictions.jsonl"
    rows = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["index"])

        if index in rows:
            raise ValueError(f"Duplicate index in {path}: {index}")

        rows[index] = row

    predictions = []

    for index in indices:
        if index not in rows:
            raise ValueError(f"Missing {mode} prediction: {index}")

        row = rows[index]
        if int(row["actual"]) != int(labels[index]):
            raise ValueError(f"Label mismatch: {index}")

        prediction = int(row["prediction"])
        if prediction not in (-1, 0, 1):
            raise ValueError(f"Invalid prediction: {prediction}")

        predictions.append(prediction)

    return np.array(predictions, dtype=np.int64)


def read_transformer(name, labels):
    path = CACHE / f"validation_{name}.npz"

    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError(f"Cache label mismatch: {name}")
        return data["probabilities"].copy()


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0
        )),
        "correct": int(np.sum(actual == predicted)),
        "errors": int(np.sum(actual != predicted))
    }


def paired_test(actual, first, second):
    first_correct = first == actual
    second_correct = second == actual

    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only

    p = (
        float(binomtest(first_only, discordant).pvalue)
        if discordant else 1.0
    )

    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "mcnemar_exact_p_value": p,
        "significant_at_0_05": bool(p < 0.05)
    }


def main():
    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(validation["label"], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    negative = rng.permutation(np.where(labels == 0)[0])
    positive = rng.permutation(np.where(labels == 1)[0])

    half = SAMPLES // 2
    indices = sorted(
        negative[:half].tolist() +
        positive[:half].tolist()
    )

    actual = labels[indices]

    zero = read_llm("zero_shot", indices, labels)
    few = read_llm("few_shot", indices, labels)

    rb_prob = read_transformer("roberta", labels)[indices]
    db_prob = read_transformer("deberta", labels)[indices]

    config = json.loads(
        (
            ROOT / "results/project07_imdb/validation_metrics.json"
        ).read_text(encoding="utf-8")
    )

    weight = float(config["selected_weight_roberta"])

    if not 0 <= weight <= 1:
        raise ValueError("Invalid ensemble weight.")

    rb = (rb_prob >= 0.5).astype(np.int64)
    db = (db_prob >= 0.5).astype(np.int64)

    ensemble = (
        weight * rb_prob + (1 - weight) * db_prob >= 0.5
    ).astype(np.int64)

    predictions = {
        "Qwen Zero-Shot": zero,
        "Qwen Few-Shot": few,
        "RoBERTa": rb,
        "DeBERTa": db,
        "Ensemble": ensemble
    }

    report = {
        "dataset": "IMDB",
        "split": "validation",
        "samples": SAMPLES,
        "seed": SEED,
        "metrics": {
            name: metrics(actual, pred)
            for name, pred in predictions.items()
        },
        "paired_tests": {
            "Few-Shot vs Zero-Shot": paired_test(
                actual, few, zero
            ),
            "Few-Shot vs Ensemble": paired_test(
                actual, few, ensemble
            ),
            "Ensemble vs RoBERTa": paired_test(
                actual, ensemble, rb
            )
        }
    }

    output = OUT / f"validation_paired_{SAMPLES}.json"
    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n===== PROJECT 08 PAIRED COMPARISON =====")

    for name, result in report["metrics"].items():
        print(
            f"{name:20s} "
            f"Accuracy={result['accuracy']:.4f} "
            f"F1={result['macro_f1']:.4f} "
            f"Errors={result['errors']}"
        )

    print("\n===== MCNEMAR TESTS =====")
    print(json.dumps(report["paired_tests"], indent=4))
    print("\nSaved:", output)
    print("Paired comparison completed successfully!")


if __name__ == "__main__":
    main()