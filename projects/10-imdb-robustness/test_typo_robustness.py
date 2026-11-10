import gc
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/project07_imdb/cache"
OUT = ROOT / "results/project10_imdb"
SAMPLES = 200
SEED = 1042

MODELS = {
    "RoBERTa": (
        ROOT / "checkpoints/project03_imdb/roberta_full"
    ),
    "DeBERTa": (
        ROOT / "checkpoints/project05_imdb/deberta_full"
    ),
}


def perturb(text, rng):
    matches = list(re.finditer(
        r"\b[A-Za-z]{6,}\b", text[:500]
    ))

    candidates = []

    for match in matches:
        word = match.group()
        if word.lower() in ("positive", "negative"):
            continue

        for pos in range(1, len(word) - 2):
            if word[pos] != word[pos + 1]:
                candidates.append((match, pos))

    if not candidates:
        raise ValueError("No suitable word for perturbation.")

    match, pos = candidates[
        int(rng.integers(len(candidates)))
    ]

    word = match.group()
    chars = list(word)

    chars[pos], chars[pos + 1] = (
        chars[pos + 1], chars[pos]
    )

    altered = (
        text[:match.start()]
        + "".join(chars)
        + text[match.end():]
    )

    if altered == text:
        raise ValueError("Perturbation did not change text.")

    return altered


def read_clean(name, labels):
    with np.load(
        CACHE / f"validation_{name.lower()}.npz",
        allow_pickle=False,
    ) as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError("Cached labels mismatch.")

        if int(data["max_length"]) != 256:
            raise ValueError("Unexpected cached max length.")

        return data["probabilities"].copy()


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
                batch,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to("cuda")

            logits = model(**inputs).logits

            probabilities.extend(
                torch.softmax(logits, dim=-1)[:, 1]
                .float().cpu().tolist()
            )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return np.asarray(probabilities)


def compare(actual, clean, noisy):
    clean_correct = clean == actual
    noisy_correct = noisy == actual

    clean_only = int(np.sum(
        clean_correct & ~noisy_correct
    ))
    noisy_only = int(np.sum(
        ~clean_correct & noisy_correct
    ))

    discordant = clean_only + noisy_only

    p = (
        float(binomtest(clean_only, discordant).pvalue)
        if discordant else 1.0
    )

    return {
        "clean_accuracy": float(
            accuracy_score(actual, clean)
        ),
        "typo_accuracy": float(
            accuracy_score(actual, noisy)
        ),
        "accuracy_change_pp": float(
            100 * (
                accuracy_score(actual, noisy)
                - accuracy_score(actual, clean)
            )
        ),
        "prediction_flips": int(
            np.sum(clean != noisy)
        ),
        "flip_rate": float(
            np.mean(clean != noisy)
        ),
        "clean_only_correct": clean_only,
        "typo_only_correct": noisy_only,
        "mcnemar_exact_p_value": p,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    dataset = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(dataset["label"], dtype=np.int64)

    rng = np.random.default_rng(SEED)
    half = SAMPLES // 2

    indices = np.sort(np.concatenate([
        rng.choice(
            np.where(labels == 0)[0],
            half, replace=False,
        ),
        rng.choice(
            np.where(labels == 1)[0],
            half, replace=False,
        ),
    ]))

    typo_rng = np.random.default_rng(2042)

    original = [
        dataset[int(i)]["text"]
        for i in indices
    ]

    altered = [
        perturb(text, typo_rng)
        for text in original
    ]

    actual = labels[indices]

    probabilities = {}
    clean_probabilities = {}

    print("GPU:", torch.cuda.get_device_name(0))
    print("Samples:", SAMPLES)

    for name, checkpoint in MODELS.items():
        print("Running:", name)

        clean_probabilities[name] = (
            read_clean(name, labels)[indices]
        )

        probabilities[name] = infer(
            checkpoint, altered
        )

    config = json.loads(
        (
            ROOT / "results/project07_imdb/validation_metrics.json"
        ).read_text(encoding="utf-8")
    )

    weight = float(config["selected_weight_roberta"])

    if not 0 <= weight <= 1:
        raise ValueError("Invalid ensemble weight.")

    clean_probabilities["Ensemble"] = (
        weight * clean_probabilities["RoBERTa"]
        + (1 - weight) * clean_probabilities["DeBERTa"]
    )

    probabilities["Ensemble"] = (
        weight * probabilities["RoBERTa"]
        + (1 - weight) * probabilities["DeBERTa"]
    )

    results = {}

    for name in ("RoBERTa", "DeBERTa", "Ensemble"):
        clean_pred = (
            clean_probabilities[name] >= 0.5
        ).astype(int)

        noisy_pred = (
            probabilities[name] >= 0.5
        ).astype(int)

        results[name] = compare(
            actual, clean_pred, noisy_pred
        )

    report = {
        "dataset": "IMDb",
        "split": "validation",
        "samples": SAMPLES,
        "seed": SEED,
        "perturbation": (
            "Swap two distinct adjacent internal letters "
            "in one word near the beginning of each review"
        ),
        "results": results,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "typo_robustness_validation_200.json"

    if output.exists():
        raise FileExistsError(
            f"Existing report preserved: {output}"
        )

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n===== PROJECT 10 TYPO ROBUSTNESS =====")

    for name, result in results.items():
        print(
            f"{name:10s} "
            f"Clean={result['clean_accuracy']:.4f} "
            f"Typo={result['typo_accuracy']:.4f} "
            f"ChangePP={result['accuracy_change_pp']:+.2f} "
            f"Flips={result['prediction_flips']} "
            f"p={result['mcnemar_exact_p_value']:.6f}"
        )

    print("\nSaved:", output)
    print("Typo robustness test completed successfully!")


if __name__ == "__main__":
    main()