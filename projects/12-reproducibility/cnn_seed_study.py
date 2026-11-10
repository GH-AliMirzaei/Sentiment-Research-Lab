import gc
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parents[2]
PROJECT09 = ROOT / "projects/09-imdb-hybrid"
OUT = ROOT / "results/project12_imdb"
BERT = ROOT / "checkpoints/project02_imdb/bert_len256_full"

SEEDS = (11, 22, 33)
TRAIN_COUNT = 512
VAL_COUNT = 200
HOLDOUT_COUNT = 1000
EPOCHS = 2


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT09 / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        row = self.dataset[self.indices[pos]]
        return row["text"], int(row["label"])


def evaluate_predictions(model, loader, device):
    model.eval()
    actual, cnn, bert = [], [], []

    with torch.inference_mode():
        for ids, mask, labels in loader:
            out, base = model(
                ids.to(device), mask.to(device)
            )
            actual.extend(labels.tolist())
            cnn.extend(out.argmax(-1).cpu().tolist())
            bert.extend(base.argmax(-1).cpu().tolist())

    return (
        np.array(actual),
        np.array(cnn),
        np.array(bert),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    if not (BERT / "config.json").exists():
        raise FileNotFoundError(BERT)

    summary_file = OUT / "cnn_seed_summary.json"

    if summary_file.exists():
        raise FileExistsError(
            "Summary already exists. Existing result preserved."
        )

    training = load_module(
        "train_bert_cnn.py", "project09_cnn"
    )
    evaluation = load_module(
        "evaluate_hybrids.py", "project09_eval"
    )

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(
        validation["label"], dtype=np.int64
    )

    indices = evaluation.get_indices(
        labels, HOLDOUT_COUNT
    )
    holdout = FixedSubset(validation, indices)

    # Fixed datasets across all three seeds.
    train = training.Reviews("train", TRAIN_COUNT, 42)
    selection = training.Reviews(
        "validation", VAL_COUNT, 43
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT), local_files_only=True
    )
    pretrained = (
        AutoModelForSequenceClassification.from_pretrained(
            str(BERT), local_files_only=True
        )
    )

    val_loader = training.make_loader(
        selection, tokenizer, False
    )
    holdout_loader = training.make_loader(
        holdout, tokenizer, False
    )

    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False

    reports = []
    baseline_reference = None

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Fixed train/selection/holdout:",
        TRAIN_COUNT, VAL_COUNT, HOLDOUT_COUNT
    )

    for seed in SEEDS:
        run_path = OUT / f"cnn_seed_{seed}.json"

        if run_path.exists():
            cached = json.loads(
                run_path.read_text(encoding="utf-8")
            )
            if cached.get("seed") != seed or not np.isclose(
                cached["holdout_1000"]["bert_accuracy"],
                0.902,
            ):
                raise ValueError(
                    f"Cached run mismatch: {run_path}"
                )

            reports.append(cached)
            print(
                f"Seed {seed}: using completed cached result"
            )
            continue

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = training.BertCNN(pretrained).to(device)

        optimizer = torch.optim.AdamW(
            (
                p for p in model.parameters()
                if p.requires_grad
            ),
            lr=2e-4,
            weight_decay=0.01,
        )

        train_loader = training.make_loader(
            train, tokenizer, True
        )

        best_f1 = -1.0
        best_epoch = None
        best_state = None

        print(f"\n===== SEED {seed} =====")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []

            for ids, mask, y in train_loader:
                optimizer.zero_grad(set_to_none=True)

                logits, _ = model(
                    ids.to(device),
                    mask.to(device),
                )

                loss = F.cross_entropy(
                    logits, y.to(device)
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    (
                        p for p in model.parameters()
                        if p.requires_grad
                    ),
                    1.0,
                )

                optimizer.step()
                losses.append(float(loss.item()))

            selection_metrics = training.evaluate(
                model, val_loader, device
            )
            f1 = selection_metrics[
                "bert_cnn"
            ]["macro_f1"]

            print(
                f"Epoch {epoch}: "
                f"loss={np.mean(losses):.4f} "
                f"val_f1={f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch

                # Only retain the small CNN head in memory.
                best_state = {
                    "convs": {
                        k: v.detach().cpu().clone()
                        for k, v in
                        model.convs.state_dict().items()
                    },
                    "classifier": {
                        k: v.detach().cpu().clone()
                        for k, v in
                        model.classifier.state_dict().items()
                    },
                }

        model.convs.load_state_dict(
            best_state["convs"]
        )
        model.classifier.load_state_dict(
            best_state["classifier"]
        )

        actual, cnn_pred, bert_pred = (
            evaluate_predictions(
                model, holdout_loader, device
            )
        )

        if baseline_reference is None:
            baseline_reference = bert_pred.copy()
        elif not np.array_equal(
            baseline_reference, bert_pred
        ):
            raise ValueError(
                "Baseline differs across runs."
            )

        bert_acc = float(
            accuracy_score(actual, bert_pred)
        )
        cnn_acc = float(
            accuracy_score(actual, cnn_pred)
        )

        if not np.isclose(bert_acc, 0.902):
            raise ValueError(
                f"Unexpected BERT baseline: {bert_acc}"
            )

        bert_ok = bert_pred == actual
        cnn_ok = cnn_pred == actual

        cnn_only = int(
            np.sum(cnn_ok & ~bert_ok)
        )
        bert_only = int(
            np.sum(~cnn_ok & bert_ok)
        )
        discordant = cnn_only + bert_only

        report = {
            "seed": seed,
            "fixed_data_seeds": {
                "train": 42,
                "validation_selection": 43,
                "holdout": 1042,
            },
            "best_epoch": best_epoch,
            "best_validation_macro_f1": float(best_f1),
            "holdout_1000": {
                "bert_accuracy": bert_acc,
                "cnn_accuracy": cnn_acc,
                "cnn_macro_f1": float(
                    f1_score(
                        actual, cnn_pred,
                        average="macro",
                    )
                ),
                "delta_percentage_points": (
                    100 * (cnn_acc - bert_acc)
                ),
                "cnn_only_correct": cnn_only,
                "bert_only_correct": bert_only,
                "mcnemar_exact_p": (
                    float(
                        binomtest(
                            cnn_only, discordant
                        ).pvalue
                    )
                    if discordant else 1.0
                ),
            },
        }

        run_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        reports.append(report)

        print(
            f"Seed {seed}: "
            f"CNN={cnn_acc:.4f} "
            f"BERT={bert_acc:.4f}"
        )

        del model, optimizer, train_loader, best_state
        gc.collect()
        torch.cuda.empty_cache()

    scores = np.asarray([
        r["holdout_1000"]["cnn_accuracy"]
        for r in reports
    ], dtype=float)

    summary = {
        "experiment": "BERT CNN seed sensitivity",
        "train_samples": TRAIN_COUNT,
        "selection_samples": VAL_COUNT,
        "disjoint_holdout_samples": HOLDOUT_COUNT,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "baseline_accuracy": 0.902,
        "mean_cnn_accuracy": float(scores.mean()),
        "sample_std_cnn_accuracy": float(
            scores.std(ddof=1)
        ),
        "min_cnn_accuracy": float(scores.min()),
        "max_cnn_accuracy": float(scores.max()),
        "runs": reports,
    }

    summary_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== PROJECT 12 SEED REPRODUCIBILITY =====")

    for result in reports:
        print(
            f"Seed={result['seed']} "
            f"BestEpoch={result['best_epoch']} "
            f"HoldoutAccuracy="
            f"{result['holdout_1000']['cnn_accuracy']:.4f}"
        )

    print(
        f"Mean={scores.mean():.4f} "
        f"SampleStd={scores.std(ddof=1):.4f}"
    )
    print("Saved:", summary_file)
    print("Seed reproducibility study completed!")


if __name__ == "__main__":
    main()