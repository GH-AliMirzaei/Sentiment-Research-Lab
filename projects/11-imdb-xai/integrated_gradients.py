import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "checkpoints/project03_imdb/roberta_full"
CACHE = ROOT / "results/project07_imdb/cache/validation_roberta.npz"
OUTPUT = ROOT / "results/project11_imdb/roberta_ig_validation_4.json"

SEED = 42
STEPS = 24
MAX_LENGTH = 256


def select_examples(labels, probabilities):
    predicted = (probabilities >= 0.5).astype(np.int64)
    rng = np.random.default_rng(SEED)

    groups = {
        "correct_negative": (labels == 0) & (predicted == 0),
        "correct_positive": (labels == 1) & (predicted == 1),
        "wrong_negative": (labels == 0) & (predicted == 1),
        "wrong_positive": (labels == 1) & (predicted == 0),
    }

    selected = []

    for name, mask in groups.items():
        candidates = np.flatnonzero(mask)

        if not len(candidates):
            raise ValueError(f"No samples for {name}")

        index = int(rng.choice(candidates))
        selected.append((name, index))

    return selected


def explain(model, tokenizer, text):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to("cuda")

    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]

    # Reference: PAD embeddings for content tokens;
    # preserve original special and padded tokens.
    special = tokenizer.get_special_tokens_mask(
        ids[0].tolist(),
        already_has_special_tokens=True,
    )

    special = torch.tensor(
        special, dtype=torch.bool, device="cuda"
    )

    reference_ids = ids.clone()
    eligible = (mask[0] == 1) & ~special

    reference_ids[0, eligible] = tokenizer.pad_token_id

    embedding = model.get_input_embeddings()

    x = embedding(ids).detach()
    baseline = embedding(reference_ids).detach()
    difference = x - baseline

    with torch.no_grad():
        original = model(
            input_ids=ids,
            attention_mask=mask,
        ).logits

        p = torch.softmax(original, dim=-1)[0]
        prediction = int(p.argmax().item())
        confidence = float(p[prediction].item())

        # Explain positive-vs-negative logit margin.
        full_margin = model(
            inputs_embeds=x,
            attention_mask=mask,
        ).logits

        base_margin = model(
            inputs_embeds=baseline,
            attention_mask=mask,
        ).logits

        margin_difference = float(
            (
                full_margin[0, 1] - full_margin[0, 0]
                - base_margin[0, 1] + base_margin[0, 0]
            ).item()
        )

    gradients = torch.zeros_like(x)

    for step in range(STEPS + 1):
        alpha = step / STEPS

        interpolated = (
            baseline + alpha * difference
        ).detach().requires_grad_(True)

        logits = model(
            inputs_embeds=interpolated,
            attention_mask=mask,
        ).logits

        margin = logits[0, 1] - logits[0, 0]

        grad = torch.autograd.grad(
            margin, interpolated
        )[0]

        factor = 0.5 if step in (0, STEPS) else 1.0
        gradients += grad.detach() * factor

    gradients /= STEPS

    scores = (
        difference * gradients
    ).sum(dim=-1)[0].detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(
        ids[0].detach().cpu().tolist()
    )

    eligible_cpu = eligible.detach().cpu().numpy()

    entries = [
        {
            "token": tokens[i],
            "score": float(scores[i]),
        }
        for i in range(len(tokens))
        if eligible_cpu[i]
    ]

    positive = sorted(
        (r for r in entries if r["score"] > 0),
        key=lambda r: r["score"],
        reverse=True,
    )[:8]

    negative = sorted(
        (r for r in entries if r["score"] < 0),
        key=lambda r: r["score"],
    )[:8]

    completeness_error = abs(
        float(scores.sum()) - margin_difference
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_positive": positive,
        "top_negative": negative,
        "completeness_error": completeness_error,
        "margin_difference": margin_difference,
        "attribution_sum": float(scores.sum()),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Existing report preserved: {OUTPUT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    dataset = load_from_disk(
        str(ROOT / "data/processed/validation")
    )

    labels = np.asarray(
        dataset["label"], dtype=np.int64
    )

    with np.load(CACHE, allow_pickle=False) as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError("Cached labels mismatch.")

        probabilities = data["probabilities"].copy()

    selected = select_examples(labels, probabilities)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH), local_files_only=True
    ).to("cuda").eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    report = {
        "project": "11",
        "method": "Integrated Gradients",
        "model": "RoBERTa",
        "split": "validation",
        "seed": SEED,
        "integration_steps": STEPS,
        "target": "positive_logit_minus_negative_logit",
        "samples": [],
    }

    print("\n===== PROJECT 11 INTEGRATED GRADIENTS =====")

    for group, index in selected:
        result = explain(
            model,
            tokenizer,
            dataset[index]["text"],
        )

        expected = int(probabilities[index] >= 0.5)

        if result["prediction"] != expected:
            raise ValueError(
                f"Prediction mismatch for index {index}"
            )

        result.update({
            "group": group,
            "index": index,
            "actual": int(labels[index]),
            "correct": (
                result["prediction"] == int(labels[index])
            ),
        })

        report["samples"].append(result)

        print(f"\n{group} | index={index}")
        print(
            f"Actual={result['actual']} "
            f"Predicted={result['prediction']} "
            f"Confidence={result['confidence']:.4f}"
        )

        print("Positive-supporting tokens:")
        print([
            item["token"]
            for item in result["top_positive"][:5]
        ])

        print("Negative-supporting tokens:")
        print([
            item["token"]
            for item in result["top_negative"][:5]
        ])

        print(
            "Completeness error:",
            round(result["completeness_error"], 5)
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT)
    print("Project 11 Integrated Gradients completed successfully!")


if __name__ == "__main__":
    main()