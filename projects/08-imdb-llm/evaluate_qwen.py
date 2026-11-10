import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/project08_imdb"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

OUT.mkdir(parents=True, exist_ok=True)


def shorten(tokenizer, text, max_tokens):
    ids = tokenizer.encode(
        text, add_special_tokens=False
    )[:max_tokens]
    return tokenizer.decode(
        ids, skip_special_tokens=True
    )


def predict(model, tokenizer, text, examples):
    messages = [{
        "role": "system",
        "content": (
            "Classify movie-review sentiment. "
            "Reply with exactly one word: positive or negative."
        )
    }]

    for example_text, label in examples:
        messages.append({
            "role": "user",
            "content": "Review:\n" + example_text
        })
        messages.append({
            "role": "assistant",
            "content": label
        })

    messages.append({
        "role": "user",
        "content": "Review:\n" + shorten(
            tokenizer, text, 256
        ) + "\nSentiment:"
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
        generated, skip_special_tokens=True
    ).strip().lower()

    match = re.match(
        r"^(positive|negative)\b", answer
    )

    prediction = (
        1 if match and match.group(1) == "positive"
        else 0 if match
        else -1
    )

    return prediction, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=20
    )
    args = parser.parse_args()

    if args.samples < 2 or args.samples % 2:
        raise ValueError(
            "Use an even sample count >= 2."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Check the GPU environment."
        )

    torch.manual_seed(42)

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    train = load_from_disk(
        str(ROOT / "data/processed/train")
    )

    labels = np.asarray(
        validation["label"], dtype=np.int64
    )

    rng = np.random.default_rng(42)

    neg = rng.permutation(
        np.where(labels == 0)[0]
    )
    pos = rng.permutation(
        np.where(labels == 1)[0]
    )

    half = args.samples // 2

    if half > min(len(neg), len(pos)):
        raise ValueError("Not enough balanced samples.")

    indices = sorted(
        neg[:half].tolist() + pos[:half].tolist()
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_ID)
    print("Validation samples:", len(indices))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    model.eval()

    train_labels = train["label"]
    train_indices = {
        0: [
            i for i, y in enumerate(train_labels)
            if y == 0
        ][:2],
        1: [
            i for i, y in enumerate(train_labels)
            if y == 1
        ][:2]
    }

    examples = []

    for j in range(2):
        for label in (0, 1):
            index = train_indices[label][j]
            examples.append((
                shorten(
                    tokenizer,
                    train[index]["text"],
                    64
                ),
                "positive" if label == 1 else "negative"
            ))

    summary = {
        "model": MODEL_ID,
        "dataset": "IMDB",
        "split": "validation",
        "samples": args.samples,
        "seed": 42,
        "results": {}
    }

    for mode in ("zero_shot", "few_shot"):
        print("\nRunning:", mode)

        predictions_file = OUT / (
            f"qwen15b_{mode}_predictions.jsonl"
        )

        saved = {}

        if predictions_file.exists():
            for line in predictions_file.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip():
                    row = json.loads(line)
                    saved[int(row["index"])] = row

        current = []

        with predictions_file.open(
            "a", encoding="utf-8"
        ) as file:

            for number, index in enumerate(indices, 1):
                actual = int(labels[index])

                if index in saved:
                    row = saved[index]
                    if row["actual"] != actual:
                        raise ValueError("Cached label mismatch.")
                else:
                    prediction, answer = predict(
                        model,
                        tokenizer,
                        validation[index]["text"],
                        examples if mode == "few_shot" else []
                    )

                    row = {
                        "index": index,
                        "actual": actual,
                        "prediction": prediction,
                        "answer": answer
                    }

                    file.write(
                        json.dumps(row) + "\n"
                    )
                    file.flush()

                current.append(row)

                print(
                    f"{mode} {number}/{len(indices)} "
                    f"actual={actual} "
                    f"predicted={row['prediction']}"
                )

        actual = np.array(
            [row["actual"] for row in current]
        )
        predicted = np.array(
            [row["prediction"] for row in current]
        )

        valid = predicted != -1

        metrics = {
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0
                )
            ),
            "valid_output_rate": float(
                np.mean(valid)
            ),
            "invalid_outputs": int(
                np.sum(~valid)
            )
        }

        summary["results"][mode] = metrics

    output = OUT / (
        f"validation_pilot_{args.samples}.json"
    )

    output.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print("\n===== PROJECT 08 RESULTS =====")
    print(json.dumps(summary, indent=4))
    print("\nProject 08 pilot completed!")


if __name__ == "__main__":
    main()