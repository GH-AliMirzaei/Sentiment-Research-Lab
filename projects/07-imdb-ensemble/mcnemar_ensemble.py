import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/project07_imdb"
CACHE = RESULTS / "cache"


def read_predictions(name):
    path = CACHE / f"test_{name}.npz"

    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"].copy()
        probabilities = data["probabilities"].copy()

    return labels, probabilities


def main():
    labels, rb = read_predictions("roberta")
    db_labels, db = read_predictions("deberta")

    if not np.array_equal(labels, db_labels):
        raise ValueError("Model labels do not match.")

    if not (len(labels) == len(rb) == len(db) == 25000):
        raise ValueError("Unexpected test data size.")

    if not (
        np.isfinite(rb).all()
        and np.isfinite(db).all()
        and np.all((rb >= 0) & (rb <= 1))
        and np.all((db >= 0) & (db <= 1))
    ):
        raise ValueError("Invalid probability values.")

    validation = json.loads(
        (RESULTS / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    weight = float(validation["selected_weight_roberta"])

    ensemble = weight * rb + (1 - weight) * db

    rb_correct = (rb >= 0.5) == labels
    en_correct = (ensemble >= 0.5) == labels

    both_correct = int(np.sum(rb_correct & en_correct))
    both_wrong = int(np.sum(~rb_correct & ~en_correct))
    rb_only = int(np.sum(rb_correct & ~en_correct))
    en_only = int(np.sum(~rb_correct & en_correct))

    discordant = rb_only + en_only

    p_value = (
        float(binomtest(rb_only, discordant).pvalue)
        if discordant else 1.0
    )

    results = {
        "baseline": "RoBERTa",
        "comparison": "RoBERTa + DeBERTa Ensemble",
        "samples": int(len(labels)),
        "selected_weight_roberta": weight,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "roberta_only_correct": rb_only,
        "ensemble_only_correct": en_only,
        "roberta_accuracy": float(np.mean(rb_correct)),
        "ensemble_accuracy": float(np.mean(en_correct)),
        "net_improvement_samples": en_only - rb_only,
        "mcnemar_exact_p_value": p_value,
        "significant_at_0_05": bool(p_value < 0.05)
    }

    test_results = json.loads(
        (RESULTS / "test_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    expected_rb = test_results["models"]["RoBERTa"]["accuracy"]
    expected_en = test_results["ensemble"]["weighted_soft_voting"]["accuracy"]

    if not np.isclose(results["roberta_accuracy"], expected_rb):
        raise ValueError("RoBERTa accuracy verification failed.")

    if not np.isclose(results["ensemble_accuracy"], expected_en):
        raise ValueError("Ensemble accuracy verification failed.")

    output = RESULTS / "mcnemar_test.json"
    output.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("\n===== PROJECT 07 MCNEMAR TEST =====")
    print(json.dumps(results, indent=4))
    print("\nProject 07 statistical analysis completed!")


if __name__ == "__main__":
    main()