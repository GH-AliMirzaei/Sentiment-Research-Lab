import gc
import json
from pathlib import Path

import numpy as np
import shap
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/project11_imdb/shap_exact_pilot.json"

MODELS = {
    "RoBERTa": ROOT / "checkpoints/project03_imdb/roberta_full",
    "DeBERTa": ROOT / "checkpoints/project05_imdb/deberta_full",
}

INDICES = [429, 2125]
GROUP_COUNT = 8
MAX_WORDS = 96
MAX_TOKENS = 256


def make_groups(review):
    words = review.split()[:MAX_WORDS]

    if len(words) < GROUP_COUNT:
        raise ValueError("Review contains too few words.")

    parts = np.array_split(np.arange(len(words)), GROUP_COUNT)

    return [
        " ".join(words[int(i)] for i in part)
        for part in parts
    ]


def explain(model, tokenizer, groups):
    def predict_margin(masks):
        masks = np.atleast_2d(masks)

        texts = [
            " ".join(
                group
                for group, active in zip(groups, mask)
                if active > 0.5
            ) or " "
            for mask in masks
        ]

        scores = []

        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                encoded = tokenizer(
                    texts[start:start + 8],
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                ).to("cuda")

                logits = model(**encoded).logits

                margin = logits[:, 1] - logits[:, 0]
                scores.extend(margin.float().cpu().tolist())

        return np.asarray(scores, dtype=np.float64)

    background = np.zeros((1, GROUP_COUNT), dtype=np.float64)
    full = np.ones((1, GROUP_COUNT), dtype=np.float64)

    explainer = shap.Explainer(
        predict_margin,
        background,
        algorithm="exact",
    )

    explanation = explainer(
        full,
        max_evals=2 ** GROUP_COUNT,
        batch_size=16,
    )

    values = np.asarray(explanation.values).reshape(-1)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    full_margin = float(predict_margin(full)[0])

    reconstruction_error = abs(
        base + float(values.sum()) - full_margin
    )

    chunks = [
        {
            "group": i + 1,
            "preview": group[:65],
            "shap_value": float(value),
        }
        for i, (group, value) in enumerate(zip(groups, values))
    ]

    return {
        "prediction": int(full_margin >= 0),
        "full_margin": full_margin,
        "baseline_margin": base,
        "reconstruction_error": reconstruction_error,
        "chunks": chunks,
        "top_positive": sorted(
            (r for r in chunks if r["shap_value"] > 0),
            key=lambda r: r["shap_value"],
            reverse=True,
        )[:3],
        "top_negative": sorted(
            (r for r in chunks if r["shap_value"] < 0),
            key=lambda r: r["shap_value"],
        )[:3],
    }


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Existing results preserved: {OUT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    dataset = load_from_disk(
        str(ROOT / "data/processed/validation")
    )

    report = {
        "project": "11",
        "method": "Exact SHAP",
        "features": "8 contiguous word groups",
        "max_words": MAX_WORDS,
        "target": "positive_logit_minus_negative_logit",
        "samples": {},
    }

    for name, path in MODELS.items():
        print("\nLoading:", name)

        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        ).to("cuda").eval()

        report["samples"][name] = {}

        for index in INDICES:
            groups = make_groups(dataset[index]["text"])

            result = explain(model, tokenizer, groups)
            result["actual"] = int(dataset[index]["label"])

            report["samples"][name][str(index)] = result

            print(f"\n{name} | Sample {index}")
            print(
                "Actual:", result["actual"],
                "Prediction:", result["prediction"]
            )
            print(
                "Reconstruction error:",
                round(result["reconstruction_error"], 6)
            )

            print("Positive-supporting groups:")
            for item in result["top_positive"]:
                print(
                    item["group"],
                    round(item["shap_value"], 4),
                    item["preview"]
                )

            print("Negative-supporting groups:")
            for item in result["top_negative"]:
                print(
                    item["group"],
                    round(item["shap_value"], 4),
                    item["preview"]
                )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n===== PROJECT 11 SHAP COMPARISON =====")
    print("Saved:", OUT)
    print("SHAP comparison completed successfully!")


if __name__ == "__main__":
    main()