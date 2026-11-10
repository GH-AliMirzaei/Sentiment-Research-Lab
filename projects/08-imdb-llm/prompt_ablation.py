import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/project08_imdb"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VARIANT = "rubric_v1"

SYSTEM_PROMPT = (
    "Classify the overall sentiment of an IMDb movie review. "
    "Positive means the reviewer has an overall favorable opinion "
    "of the movie. Negative means an overall unfavorable opinion. "
    "Pay attention to negation, sarcasm, contrast, and mixed opinions. "
    "A sad story does not necessarily mean a negative movie review. "
    "When both praise and criticism appear, identify the reviewer's "
    "overall evaluation of the movie. "
    "Reply with exactly one lowercase word: positive or negative. "
    "Do not provide an explanation."
)


def read_rows(path):
    rows = {}

    if not path.exists():
        return rows

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        index = int(row["index"])

        if index in rows:
            raise ValueError(f"Duplicate prediction index: {index}")

        rows[index] = row

    return rows


def shorten(tokenizer, text, limit):
    ids = tokenizer.encode(
        text, add_special_tokens=False
    )[:limit]

    return tokenizer.decode(
        ids, skip_special_tokens=True
    )


def predict(model, tokenizer, review, examples):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for text, label in examples:
        messages.append({
            "role": "user",
            "content": "Review:\n" + text
        })
        messages.append({
            "role": "assistant",
            "content": label
        })

    messages.append({
        "role": "user",
        "content": (
            "Review:\n"
            + shorten(tokenizer, review, 256)
            + "\nSentiment:"
        )
    })

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip().lower()

    match = re.match(r"^(positive|negative)\b", answer)

    prediction = (
        1 if match and match.group(1) == "positive"
        else 0 if match
        else -1
    )

    return prediction, answer


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
        "errors": int(np.sum(actual != predicted)),
        "invalid_outputs": int(np.sum(predicted == -1))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    if not 2 <= args.samples <= 5000 or args.samples % 2:
        raise ValueError("Samples must be even, between 2 and 5000.")

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    train = load_from_disk(
        str(ROOT / "data/processed/train")
    )

    labels = np.asarray(validation["label"], dtype=np.int64)

    rng = np.random.default_rng(42)
    neg = rng.permutation(np.where(labels == 0)[0])
    pos = rng.permutation(np.where(labels == 1)[0])

    half = args.samples // 2
    indices = sorted(
        neg[:half].tolist() + pos[:half].tolist()
    )

    baseline_path = OUT / "qwen15b_few_shot_predictions.jsonl"
    new_path = OUT / f"qwen15b_{VARIANT}_predictions.jsonl"

    baseline = read_rows(baseline_path)
    existing = read_rows(new_path)

    for index in indices:
        if index not in baseline:
            raise ValueError(
                f"Missing original Few-Shot prediction: {index}"
            )

        if int(baseline[index]["actual"]) != int(labels[index]):
            raise ValueError("Baseline label mismatch.")

        if int(baseline[index]["prediction"]) not in (-1, 0, 1):
            raise ValueError("Invalid baseline prediction.")

        if index in existing:
            row = existing[index]
            if int(row["actual"]) != int(labels[index]):
                raise ValueError("Cached rubric label mismatch.")
            if row.get("variant") != VARIANT:
                raise ValueError("Cached prompt variant mismatch.")

    pending = [index for index in indices if index not in existing]

    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")

        print("GPU:", torch.cuda.get_device_name(0))
        print("Model:", MODEL_ID)
        print("Pending predictions:", len(pending))

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

        model.eval()

        train_labels = train["label"]
        example_indices = {
            0: [i for i, y in enumerate(train_labels) if y == 0][:2],
            1: [i for i, y in enumerate(train_labels) if y == 1][:2]
        }

        examples = []

        for j in range(2):
            for label in (0, 1):
                i = example_indices[label][j]
                examples.append((
                    shorten(tokenizer, train[i]["text"], 64),
                    "positive" if label == 1 else "negative"
                ))

        with new_path.open("a", encoding="utf-8") as file:
            for number, index in enumerate(pending, 1):
                prediction, answer = predict(
                    model,
                    tokenizer,
                    validation[index]["text"],
                    examples
                )

                row = {
                    "index": int(index),
                    "actual": int(labels[index]),
                    "prediction": int(prediction),
                    "answer": answer,
                    "variant": VARIANT
                }

                file.write(json.dumps(row) + "\n")
                file.flush()
                existing[index] = row

                print(
                    f"{VARIANT} {number}/{len(pending)} "
                    f"actual={row['actual']} "
                    f"predicted={prediction}"
                )

    actual = np.asarray(
        [labels[i] for i in indices], dtype=np.int64
    )
    base_pred = np.asarray(
        [baseline[i]["prediction"] for i in indices],
        dtype=np.int64
    )
    new_pred = np.asarray(
        [existing[i]["prediction"] for i in indices],
        dtype=np.int64
    )

    base_correct = base_pred == actual
    new_correct = new_pred == actual

    base_only = int(np.sum(base_correct & ~new_correct))
    new_only = int(np.sum(~base_correct & new_correct))

    discordant = base_only + new_only

    p_value = (
        float(binomtest(base_only, discordant).pvalue)
        if discordant else 1.0
    )

    result = {
        "model": MODEL_ID,
        "split": "validation",
        "samples": args.samples,
        "seed": 42,
        "prompt_variant": VARIANT,
        "system_prompt": SYSTEM_PROMPT,
        "original_few_shot": metrics(actual, base_pred),
        "rubric_few_shot": metrics(actual, new_pred),
        "mcnemar": {
            "original_only_correct": base_only,
            "rubric_only_correct": new_only,
            "discordant": discordant,
            "exact_p_value": p_value,
            "significant_at_0_05": bool(p_value < 0.05)
        }
    }

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"validation_prompt_ablation_{args.samples}.json"

    output.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print("\n===== PROJECT 08 PROMPT ABLATION =====")
    print(json.dumps(result, indent=4))
    print("\nPrompt experiment completed successfully!")


if __name__ == "__main__":
    main()