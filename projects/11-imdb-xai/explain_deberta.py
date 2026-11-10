import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "projects/11-imdb-xai/integrated_gradients.py"
MODEL = ROOT / "checkpoints/project05_imdb/deberta_full"
CACHE = ROOT / "results/project07_imdb/cache/validation_deberta.npz"
REFERENCE = ROOT / "results/project11_imdb/roberta_ig_validation_4_steps256.json"
OUTPUT = ROOT / "results/project11_imdb/deberta_ig_same4_steps96.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Results already exist: {OUTPUT}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    spec = importlib.util.spec_from_file_location("original_ig", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.STEPS = 96

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    selected = [
        (row["group"], int(row["index"]))
        for row in reference["samples"]
    ]

    if len(selected) != 4 or len(set(i for _, i in selected)) != 4:
        raise ValueError("Unexpected reference samples.")

    dataset = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(dataset["label"], dtype=np.int64)

    with np.load(CACHE, allow_pickle=False) as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError("DeBERTa cache label mismatch.")
        probabilities = data["probabilities"].copy()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL), local_files_only=True
    ).to("cuda").eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    report = {
        "project": "11",
        "model": "DeBERTa-v3-small",
        "method": "Integrated Gradients",
        "integration_steps": 96,
        "target": "positive_logit_minus_negative_logit",
        "selected_using": "RoBERTa reference predictions",
        "samples": [],
    }

    print("\n===== PROJECT 11 DEBERTA XAI =====")

    for original_group, index in selected:
        result = module.explain(
            model, tokenizer, dataset[index]["text"]
        )

        expected = int(probabilities[index] >= 0.5)

        if result["prediction"] != expected:
            raise ValueError(
                f"Prediction mismatch for index {index}"
            )

        result["index"] = index
        result["roberta_reference_group"] = original_group
        result["actual"] = int(labels[index])
        result["correct"] = (
            result["prediction"] == int(labels[index])
        )

        relative_error = (
            result["completeness_error"]
            / max(abs(result["margin_difference"]), 1e-8)
        )
        result["relative_completeness_error"] = relative_error

        report["samples"].append(result)

        print(f"\nSample: {index}")
        print("Actual:", result["actual"])
        print("DeBERTa prediction:", result["prediction"])
        print("Confidence:", round(result["confidence"], 4))
        print("Correct:", result["correct"])
        print(
            "Positive tokens:",
            [x["token"] for x in result["top_positive"][:5]]
        )
        print(
            "Negative tokens:",
            [x["token"] for x in result["top_negative"][:5]]
        )
        print(
            "Relative completeness error:",
            f"{100 * relative_error:.2f}%"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nSaved:", OUTPUT)
    print("DeBERTa XAI completed successfully!")


if __name__ == "__main__":
    main()