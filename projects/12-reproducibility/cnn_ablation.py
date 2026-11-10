import gc
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
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
VARIANTS = {
    "without_kernel_2": (0, 2),
    "without_kernel_3": (1, 3),
    "without_kernel_4": (2, 4),
}
EPOCHS = 2


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.dataset[self.indices[i]]
        return row["text"], int(row["label"])


def suppress_branch(module, inputs, output):
    return torch.zeros_like(output)


def predict(model, loader, device):
    model.eval()
    actual, predictions, baseline = [], [], []

    with torch.inference_mode():
        for ids, mask, labels in loader:
            logits, bert_logits = model(
                ids.to(device), mask.to(device)
            )
            actual.extend(labels.tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
            baseline.extend(bert_logits.argmax(-1).cpu().tolist())

    return (
        np.asarray(actual),
        np.asarray(predictions),
        np.asarray(baseline),
    )


def train_variant(
    seed, variant, branch_index, kernel,
    training, pretrained, train_data,
    tokenizer, selection_loader, holdout_loader,
    device,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = training.BertCNN(pretrained).to(device)

    kernels = [
        int(conv.kernel_size[0])
        for conv in model.convs
    ]

    if kernels != [2, 3, 4]:
        raise ValueError(
            f"Unexpected CNN architecture: {kernels}"
        )

    handle = model.convs[branch_index].register_forward_hook(
        suppress_branch
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=2e-4,
        weight_decay=0.01,
    )

    train_loader = training.make_loader(
        train_data, tokenizer, True
    )

    best_f1 = -1.0
    best_epoch = None
    best_state = None

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []

            for ids, mask, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)

                logits, _ = model(
                    ids.to(device), mask.to(device)
                )

                loss = F.cross_entropy(
                    logits, labels.to(device)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    1.0,
                )

                optimizer.step()
                losses.append(float(loss.item()))

            selection_metrics = training.evaluate(
                model, selection_loader, device
            )

            score = float(
                selection_metrics["bert_cnn"]["macro_f1"]
            )

            print(
                f"Epoch {epoch}: "
                f"Loss={np.mean(losses):.4f} "
                f"SelectionF1={score:.4f}",
                flush=True,
            )

            if score > best_f1:
                best_f1 = score
                best_epoch = epoch

                best_state = {
                    "convs": {
                        k: v.detach().cpu().clone()
                        for k, v in model.convs.state_dict().items()
                    },
                    "classifier": {
                        k: v.detach().cpu().clone()
                        for k, v in model.classifier.state_dict().items()
                    },
                }

        model.convs.load_state_dict(best_state["convs"])
        model.classifier.load_state_dict(
            best_state["classifier"]
        )

        actual, predicted, baseline = predict(
            model, holdout_loader, device
        )

        baseline_accuracy = float(
            accuracy_score(actual, baseline)
        )
        accuracy = float(
            accuracy_score(actual, predicted)
        )

        if not np.isclose(baseline_accuracy, 0.902):
            raise ValueError(
                f"Unexpected BERT baseline: {baseline_accuracy}"
            )

        return {
            "variant": variant,
            "disabled_kernel": kernel,
            "active_kernels": [
                k for k in (2, 3, 4) if k != kernel
            ],
            "seed": seed,
            "best_epoch": best_epoch,
            "selection_macro_f1": best_f1,
            "holdout_accuracy": accuracy,
            "holdout_macro_f1": float(
                f1_score(actual, predicted, average="macro")
            ),
            "bert_accuracy": baseline_accuracy,
            "holdout_samples": len(actual),
        }

    finally:
        handle.remove()
        del model, optimizer, train_loader, best_state
        gc.collect()
        torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")

    summary_path = OUT / "cnn_seed_summary.json"
    full_study = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    full = {
        int(r["seed"]): float(
            r["holdout_1000"]["cnn_accuracy"]
        )
        for r in full_study["runs"]
    }

    if set(full) != set(SEEDS):
        raise ValueError("Missing full-CNN reference runs.")

    training = import_script(
        PROJECT09 / "train_bert_cnn.py",
        "ablation_training",
    )
    evaluation = import_script(
        PROJECT09 / "evaluate_hybrids.py",
        "ablation_evaluation",
    )

    validation = load_from_disk(
        str(ROOT / "data/processed/validation")
    )
    labels = np.asarray(validation["label"], dtype=np.int64)

    indices = evaluation.get_indices(labels, 1000)

    train_data = training.Reviews("train", 512, 42)
    selection = training.Reviews("validation", 200, 43)
    holdout = FixedSubset(validation, indices)

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT), local_files_only=True
    )

    pretrained = AutoModelForSequenceClassification.from_pretrained(
        str(BERT), local_files_only=True
    )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    selection_loader = training.make_loader(
        selection, tokenizer, False
    )
    holdout_loader = training.make_loader(
        holdout, tokenizer, False
    )

    OUT.mkdir(parents=True, exist_ok=True)

    reports = []

    print("\n===== PROJECT 12 CNN ABLATION =====")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Full CNN reference:", full)

    for variant, (branch_index, kernel) in VARIANTS.items():
        for seed in SEEDS:
            path = OUT / f"ablation_{variant}_seed{seed}.json"

            print(
                f"\nVariant={variant} | Seed={seed}",
                flush=True,
            )

            if path.exists():
                result = json.loads(
                    path.read_text(encoding="utf-8")
                )

                if (
                    result.get("seed") != seed
                    or result.get("variant") != variant
                    or result.get("disabled_kernel") != kernel
                ):
                    raise ValueError(
                        f"Cached run mismatch: {path}"
                    )

                print("Using completed cached result.")
            else:
                result = train_variant(
                    seed, variant, branch_index, kernel,
                    training, pretrained, train_data,
                    tokenizer, selection_loader,
                    holdout_loader, device,
                )

                path.write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

            result["full_accuracy_same_seed"] = full[seed]
            result["delta_vs_full_pp"] = round(
                100 * (
                    result["holdout_accuracy"] - full[seed]
                ),
                4,
            )

            reports.append(result)

            print(
                f"Ablated={result['holdout_accuracy']:.4f} "
                f"Full={full[seed]:.4f} "
                f"DeltaPP={result['delta_vs_full_pp']:+.2f}",
                flush=True,
            )

    comparison = {}

    for variant in VARIANTS:
        subset = [
            r for r in reports
            if r["variant"] == variant
        ]
        scores = np.asarray(
            [r["holdout_accuracy"] for r in subset]
        )
        deltas = np.asarray(
            [r["delta_vs_full_pp"] for r in subset]
        )

        comparison[variant] = {
            "mean_accuracy": float(scores.mean()),
            "sample_std_accuracy": float(
                scores.std(ddof=1)
            ),
            "mean_delta_vs_full_pp": float(deltas.mean()),
            "seed_results": [
                {
                    "seed": r["seed"],
                    "accuracy": r["holdout_accuracy"],
                    "delta_vs_full_pp": r["delta_vs_full_pp"],
                }
                for r in subset
            ],
        }

    result = {
        "project": "12",
        "experiment": "Retrained functional CNN branch ablation",
        "description": (
            "One convolution branch is suppressed during "
            "training and evaluation using a forward hook."
        ),
        "active_reference_kernels": [2, 3, 4],
        "seeds": list(SEEDS),
        "train_samples": 512,
        "selection_samples": 200,
        "holdout_samples": 1000,
        "epochs": EPOCHS,
        "full_mean_accuracy": float(
            np.mean(list(full.values()))
        ),
        "comparison": comparison,
        "runs": reports,
        "caution": (
            "Descriptive differences only; full-versus-ablated "
            "paired per-sample predictions were not retained."
        ),
    }

    output = OUT / "cnn_ablation_summary.json"

    if output.exists():
        raise FileExistsError(
            f"Existing summary preserved: {output}"
        )

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n===== PROJECT 12 ABLATION RESULTS =====")
    print(
        f"Full CNN mean accuracy: "
        f"{result['full_mean_accuracy']:.4f}"
    )

    for name, row in comparison.items():
        print(
            f"{name:20s} "
            f"Mean={row['mean_accuracy']:.4f} "
            f"DeltaPP={row['mean_delta_vs_full_pp']:+.3f} "
            f"Std={row['sample_std_accuracy']:.4f}"
        )

    print("\nSaved:", output)
    print("Project 12 ablation completed successfully!")


if __name__ == "__main__":
    main()