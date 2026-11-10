import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "projects/09-imdb-hybrid"
BERT = ROOT / "checkpoints/project02_imdb/bert_len256_full"
CHECKPOINTS = ROOT / "checkpoints/project09_imdb"
RESULTS = ROOT / "results/project09_imdb"

CNN_HEAD = CHECKPOINTS / "bert_cnn_train512_val200_epochs2.pt"
LSTM_HEAD = CHECKPOINTS / "bert_bilstm_train512_val200_epochs2.pt"


def load_class(filename, module_name, class_name):
    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_head(path, module, layer_name):
    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(
        path, map_location="cpu", weights_only=True
    )

    expected = str(BERT.relative_to(ROOT))

    if data["bert_source"] != expected:
        raise ValueError("BERT checkpoint mismatch.")

    if data["max_length"] != 256:
        raise ValueError("Input length mismatch.")

    getattr(module, layer_name).load_state_dict(
        data[layer_name]
    )
    module.classifier.load_state_dict(
        data["classifier"]
    )


def get_indices(labels, count):
    if count < 2 or count > 4800 or count % 2:
        raise ValueError(
            "Samples must be even and between 2 and 4800."
        )

    negatives = np.where(labels == 0)[0]
    positives = np.where(labels == 1)[0]

    # Reproduce the original 200-sample epoch-selection set.
    old_rng = np.random.default_rng(43)
    old_neg = old_rng.choice(
        negatives, 100, replace=False
    )
    old_pos = old_rng.choice(
        positives, 100, replace=False
    )

    old_indices = np.concatenate([old_neg, old_pos])

    available_neg = np.setdiff1d(negatives, old_indices)
    available_pos = np.setdiff1d(positives, old_indices)

    rng = np.random.default_rng(1042)
    half = count // 2

    chosen = np.concatenate([
        rng.choice(available_neg, half, replace=False),
        rng.choice(available_pos, half, replace=False),
    ])

    assert not np.isin(chosen, old_indices).any()

    return np.sort(chosen)


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted, average="macro"
        )),
        "correct": int(np.sum(actual == predicted)),
        "errors": int(np.sum(actual != predicted)),
    }


def mcnemar(actual, first, second):
    first_correct = first == actual
    second_correct = second == actual

    first_only = int(np.sum(
        first_correct & ~second_correct
    ))
    second_only = int(np.sum(
        ~first_correct & second_correct
    ))

    discordant = first_only + second_only

    p = (
        float(binomtest(first_only, discordant).pvalue)
        if discordant else 1.0
    )

    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant": discordant,
        "exact_p_value": p,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )

    labels = np.asarray(
        validation["label"], dtype=np.int64
    )

    indices = get_indices(labels, args.samples)
    actual = labels[indices]

    print("GPU:", torch.cuda.get_device_name(0))
    print("Evaluation samples:", len(indices))
    print("Overlap with original Validation-200: 0")

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT), local_files_only=True
    )

    pretrained = (
        AutoModelForSequenceClassification.from_pretrained(
            str(BERT), local_files_only=True
        )
    )

    CNN = load_class(
        "train_bert_cnn.py", "hybrid_cnn", "BertCNN"
    )
    BiLSTM = load_class(
        "train_bert_bilstm.py", "hybrid_lstm", "BertBiLSTM"
    )

    cnn = CNN(pretrained)
    bilstm = BiLSTM(pretrained)

    load_head(CNN_HEAD, cnn, "convs")
    load_head(LSTM_HEAD, bilstm, "lstm")

    cnn = cnn.to("cuda").eval()
    bilstm = bilstm.to("cuda").eval()

    results = {
        "BERT": [],
        "BERT_CNN": [],
        "BERT_BiLSTM": [],
    }

    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            batch_indices = indices[start:start + 8]

            texts = [
                validation[int(i)]["text"]
                for i in batch_indices
            ]

            encoded = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            ).to("cuda")

            cnn_logits, bert_logits = cnn(
                encoded["input_ids"],
                encoded["attention_mask"],
            )

            lstm_logits, other_bert_logits = bilstm(
                encoded["input_ids"],
                encoded["attention_mask"],
            )

            if not torch.allclose(
                bert_logits, other_bert_logits, atol=1e-5
            ):
                raise ValueError("Baseline predictions differ.")

            for key, logits in (
                ("BERT", bert_logits),
                ("BERT_CNN", cnn_logits),
                ("BERT_BiLSTM", lstm_logits),
            ):
                results[key].extend(
                    logits.argmax(dim=-1).cpu().tolist()
                )

            if (start // 8 + 1) % 25 == 0:
                print(f"Evaluated {start + len(batch_indices)}/{len(indices)}")

    predictions = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in results.items()
    }

    pairs = {
        "CNN_vs_BERT": ("BERT_CNN", "BERT"),
        "BiLSTM_vs_BERT": ("BERT_BiLSTM", "BERT"),
        "CNN_vs_BiLSTM": ("BERT_CNN", "BERT_BiLSTM"),
    }

    paired = {
        name: mcnemar(
            actual,
            predictions[first],
            predictions[second],
        )
        for name, (first, second) in pairs.items()
    }

    # Holm correction across three comparisons.
    ordered = sorted(
        paired,
        key=lambda key: paired[key]["exact_p_value"],
    )

    previous = 0.0
    for position, name in enumerate(ordered):
        adjusted = min(
            1.0,
            (len(ordered) - position)
            * paired[name]["exact_p_value"],
        )
        previous = max(previous, adjusted)
        paired[name]["holm_adjusted_p_value"] = previous

    report = {
        "dataset": "IMDB",
        "split": "validation_disjoint_holdout",
        "samples": len(indices),
        "seed": 1042,
        "model_selection_overlap": 0,
        "metrics": {
            name: metrics(actual, prediction)
            for name, prediction in predictions.items()
        },
        "paired_tests": paired,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)

    output = RESULTS / (
        f"hybrid_holdout_{args.samples}.json"
    )

    if output.exists():
        raise FileExistsError(
            f"Existing results preserved: {output}"
        )

    output.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n===== PROJECT 09 HYBRID COMPARISON =====")
    print(json.dumps(report, indent=4))
    print("\nHybrid evaluation completed successfully!")


if __name__ == "__main__":
    main()