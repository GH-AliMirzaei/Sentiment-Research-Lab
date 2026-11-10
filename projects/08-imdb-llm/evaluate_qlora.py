import argparse
import json
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = ROOT / "checkpoints/project08_imdb/qwen_qlora_2000"
OUT = ROOT / "results/project08_imdb"
OUT.mkdir(parents=True, exist_ok=True)


def predict(model, tokenizer, review):
    review_ids = tokenizer.encode(
        review, add_special_tokens=False
    )[:160]

    short_review = tokenizer.decode(
        review_ids, skip_special_tokens=True
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Classify movie-review sentiment. "
                "Reply with exactly one word: "
                "positive or negative."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review:\n" + short_review + "\nSentiment:"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated, skip_special_tokens=True
    ).strip().lower()

    match = re.match(r"^(positive|negative)\b", answer)

    if not match:
        return -1, answer

    return (1 if match.group(1) == "positive" else 0), answer


def read_cache(path):
    rows = {}

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"Duplicate index: {index}")
            rows[index] = row

    return rows


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )),
        "correct": int(np.sum(actual == predicted)),
        "errors": int(np.sum(actual != predicted)),
        "invalid_outputs": int(np.sum(predicted == -1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    if args.samples < 2 or args.samples > 5000 or args.samples % 2:
        raise ValueError("Samples must be even, between 2 and 5000.")

    if not ADAPTER.joinpath("adapter_config.json").exists():
        raise FileNotFoundError("Trained QLoRA adapter not found.")

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(validation["label"], dtype=np.int64)

    rng = np.random.default_rng(42)
    negative = rng.permutation(np.where(labels == 0)[0])
    positive = rng.permutation(np.where(labels == 1)[0])

    half = args.samples // 2
    indices = sorted(
        negative[:half].tolist() + positive[:half].tolist()
    )

    modes = ("base_4bit", "qlora_2000")
    paths = {
        mode: OUT / f"validation_qlora_{mode}_predictions.jsonl"
        for mode in modes
    }
    caches = {mode: read_cache(paths[mode]) for mode in modes}

    for mode in modes:
        for index in indices:
            if index in caches[mode]:
                if int(caches[mode][index]["actual"]) != int(labels[index]):
                    raise ValueError("Cached label mismatch.")

    pending = {
        mode: [i for i in indices if i not in caches[mode]]
        for mode in modes
    }

    if any(pending.values()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable.")

        dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, local_files_only=True
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            local_files_only=True,
            quantization_config=quantization,
            dtype=dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(
            base, str(ADAPTER), is_trainable=False
        )
        model.eval()

        print("GPU:", torch.cuda.get_device_name(0))
        print("Adapter:", ADAPTER)

        for mode in modes:
            todo = pending[mode]
            print(f"\nRunning {mode}: {len(todo)} pending samples")

            context = (
                model.disable_adapter()
                if mode == "base_4bit"
                else nullcontext()
            )

            with context:
                with paths[mode].open("a", encoding="utf-8") as file:
                    for number, index in enumerate(todo, 1):
                        prediction, answer = predict(
                            model,
                            tokenizer,
                            validation[index]["text"],
                        )

                        row = {
                            "index": int(index),
                            "actual": int(labels[index]),
                            "prediction": int(prediction),
                            "answer": answer,
                        }

                        file.write(json.dumps(row) + "\n")
                        file.flush()
                        caches[mode][index] = row

                        if number % 20 == 0 or number == len(todo):
                            print(f"{mode}: {number}/{len(todo)}")

    actual = labels[indices]
    predictions = {}

    for mode in modes:
        predictions[mode] = np.asarray([
            int(caches[mode][i]["prediction"])
            for i in indices
        ], dtype=np.int64)

    base_correct = predictions["base_4bit"] == actual
    tuned_correct = predictions["qlora_2000"] == actual

    base_only = int(np.sum(base_correct & ~tuned_correct))
    tuned_only = int(np.sum(~base_correct & tuned_correct))
    discordant = base_only + tuned_only

    p_value = (
        float(binomtest(base_only, discordant).pvalue)
        if discordant else 1.0
    )

    report = {
        "dataset": "IMDB",
        "split": "validation",
        "samples": args.samples,
        "seed": 42,
        "evaluation_prompt": "Original training prompt, 160 review tokens",
        "base_4bit": metrics(actual, predictions["base_4bit"]),
        "qlora_2000": metrics(actual, predictions["qlora_2000"]),
        "mcnemar": {
            "base_only_correct": base_only,
            "qlora_only_correct": tuned_only,
            "discordant": discordant,
            "p_value": p_value,
            "significant_at_0_05": bool(p_value < 0.05),
        },
    }

    output = OUT / f"validation_qlora_comparison_{args.samples}.json"
    output.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n===== PROJECT 08 QLORA EVALUATION =====")
    print(json.dumps(report, indent=4))
    print("\nQLoRA evaluation completed successfully!")


if __name__ == "__main__":
    main()