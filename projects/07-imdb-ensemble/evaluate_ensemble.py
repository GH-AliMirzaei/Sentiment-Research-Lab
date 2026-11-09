import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/project07_imdb"
CACHE = RESULTS / "cache"

MODELS = {
    "RoBERTa": ROOT / "checkpoints/project03_imdb/roberta_full",
    "DeBERTa": ROOT / "checkpoints/project05_imdb/deberta_full"
}

BATCH_SIZE = 8
MAX_LENGTH = 256
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


def get_probabilities(name, checkpoint, dataset, split):
    cache_file = CACHE / f"{split}_{name.lower()}.npz"
    labels = np.asarray(dataset["label"], dtype=np.int64)

    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cached:
            valid = (
                np.array_equal(cached["labels"], labels)
                and int(cached["max_length"]) == MAX_LENGTH
                and str(cached["checkpoint"].item())
                == str(checkpoint.resolve())
            )
            if not valid:
                raise ValueError(f"Stale or incompatible cache: {cache_file}")

            print(f"Using cached predictions: {name}, {split}")
            return cached["probabilities"].copy()

    print(f"\nRunning {name} on {split}...")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint)
    ).to(DEVICE)
    model.eval()

    if model.config.num_labels != 2:
        raise ValueError(f"{name} must have exactly 2 classes.")

    probabilities = []

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(dataset), BATCH_SIZE),
            desc=f"{name} - {split}"
        ):
            texts = dataset[start:start + BATCH_SIZE]["text"]

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

            output = model(**inputs)
            prob = torch.softmax(output.logits, dim=-1)[:, 1]

            probabilities.extend(prob.cpu().tolist())

    result = np.asarray(probabilities, dtype=np.float32)

    temporary = cache_file.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            labels=labels,
            probabilities=result,
            max_length=np.array(MAX_LENGTH),
            checkpoint=np.array(str(checkpoint.resolve()))
        )
    temporary.replace(cache_file)

    del model, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def calculate_metrics(labels, probabilities):
    predicted = (probabilities >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_macro": float(
            precision_score(labels, predicted, average="macro")
        ),
        "recall_macro": float(
            recall_score(labels, predicted, average="macro")
        ),
        "f1_macro": float(
            f1_score(labels, predicted, average="macro")
        ),
        "errors": int(np.sum(predicted != labels)),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=[0, 1]
        ).tolist()
    }


def hard_vote(p_rb, p_db):
    rb = p_rb >= 0.5
    db = p_db >= 0.5

    # Two models can tie. Break ties using the model
    # farther from probability 0.5 (higher confidence).
    choose_rb = np.abs(p_rb - 0.5) >= np.abs(p_db - 0.5)
    final = np.where(rb == db, rb, np.where(choose_rb, rb, db))

    return final.astype(np.float32)


def save_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["validation", "test"],
        required=True
    )
    args = parser.parse_args()

    validation_file = RESULTS / "validation_metrics.json"

    if args.stage == "test" and not validation_file.exists():
        raise FileNotFoundError(
            "Run --stage validation before evaluating test."
        )

    for name, path in MODELS.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {path}")

    print("Device:", DEVICE)
    print("Stage:", args.stage)

    dataset = load_from_disk(
        str(ROOT / "data/processed" / args.stage)
    )
    labels = np.asarray(dataset["label"], dtype=np.int64)

    rb = get_probabilities(
        "RoBERTa", MODELS["RoBERTa"], dataset, args.stage
    )
    db = get_probabilities(
        "DeBERTa", MODELS["DeBERTa"], dataset, args.stage
    )

    hard = hard_vote(rb, db)
    soft = 0.5 * rb + 0.5 * db

    if args.stage == "validation":
        best_weight = 0.5
        best_score = -1.0
        grid = []

        for i in range(11):
            weight = i / 10
            combined = weight * rb + (1 - weight) * db
            score = calculate_metrics(
                labels, combined
            )["f1_macro"]

            grid.append({
                "weight_roberta": weight,
                "f1_macro": score
            })

            if (
                score > best_score + 1e-12
                or (
                    abs(score - best_score) <= 1e-12
                    and abs(weight - 0.5)
                    < abs(best_weight - 0.5)
                )
            ):
                best_score = score
                best_weight = weight

        print("\nSelected RoBERTa weight:", best_weight)

    else:
        config = json.loads(
            validation_file.read_text(encoding="utf-8")
        )
        best_weight = float(config["selected_weight_roberta"])

    weighted = best_weight * rb + (1 - best_weight) * db

    results = {
        "dataset": "IMDB",
        "split": args.stage,
        "samples": len(labels),
        "max_length": MAX_LENGTH,
        "selected_weight_roberta": best_weight,
        "selected_weight_deberta": 1 - best_weight,
        "selection_metric": "validation_macro_f1",
        "models": {
            "RoBERTa": calculate_metrics(labels, rb),
            "DeBERTa": calculate_metrics(labels, db)
        },
        "ensemble": {
            "hard_voting_confidence_tiebreak": calculate_metrics(
                labels, hard
            ),
            "soft_voting": calculate_metrics(labels, soft),
            "weighted_soft_voting": calculate_metrics(
                labels, weighted
            )
        }
    }

    if args.stage == "validation":
        results["weight_search"] = grid

    output = RESULTS / f"{args.stage}_metrics.json"
    save_json(output, results)

    print("\n===== PROJECT 07 RESULTS =====")
    print("Split:", args.stage)
    print("RoBERTa weight:", best_weight)

    for name, metrics in {
        **results["models"],
        **results["ensemble"]
    }.items():
        print(
            f"{name:35s} "
            f"Accuracy={metrics['accuracy']:.6f} "
            f"Macro-F1={metrics['f1_macro']:.6f}"
        )

    print("\nSaved:", output)
    print("Project 07 stage completed successfully!")


if __name__ == "__main__":
    main()