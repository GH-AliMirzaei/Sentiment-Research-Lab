import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[2]
BERT_PATH = ROOT / "checkpoints/project02_imdb/bert_len256_full"
RESULTS = ROOT / "results/project09_imdb"
CHECKPOINTS = ROOT / "checkpoints/project09_imdb"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8


class Reviews(Dataset):
    def __init__(self, split, count, seed):
        data = load_from_disk(str(ROOT / "data/processed" / split))
        labels = np.asarray(data["label"], dtype=np.int64)

        if count > len(data) or count < 2 or count % 2:
            raise ValueError(f"Invalid sample count for {split}")

        rng = np.random.default_rng(seed)
        half = count // 2

        neg = rng.choice(np.where(labels == 0)[0], half, replace=False)
        pos = rng.choice(np.where(labels == 1)[0], half, replace=False)
        indices = rng.permutation(np.concatenate([neg, pos]))

        self.rows = [data[int(i)] for i in indices]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return row["text"], int(row["label"])


def make_loader(dataset, tokenizer, shuffle):
    def collate(items):
        texts, labels = zip(*items)
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate,
    )


class BertCNN(nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        if pretrained.config.model_type != "bert":
            raise ValueError("Expected a BERT checkpoint.")

        for parameter in pretrained.parameters():
            parameter.requires_grad = False

        self.encoder = pretrained.bert
        self.baseline_dropout = pretrained.dropout
        self.baseline_classifier = pretrained.classifier

        hidden_size = pretrained.config.hidden_size
        self.kernel_sizes = (2, 3, 4)

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, 96, kernel_size=k)
            for k in self.kernel_sizes
        ])

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(96 * len(self.kernel_sizes), 2)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.baseline_dropout.eval()
        self.baseline_classifier.eval()
        return self

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            original_logits = self.baseline_classifier(
                self.baseline_dropout(bert_output.pooler_output)
            )

        hidden = bert_output.last_hidden_state.transpose(1, 2)
        attention = attention_mask.unsqueeze(1).float()

        features = []

        for kernel, conv in zip(self.kernel_sizes, self.convs):
            values = F.relu(conv(hidden))

            valid = F.avg_pool1d(
                attention, kernel_size=kernel, stride=1
            ) == 1

            values = values.masked_fill(~valid, -1e4)
            pooled = values.amax(dim=2)
            features.append(pooled)

        combined = torch.cat(features, dim=1)
        logits = self.classifier(self.dropout(combined))

        return logits, original_logits


def evaluate(model, loader, device):
    model.eval()

    actual = []
    cnn_predictions = []
    bert_predictions = []

    with torch.inference_mode():
        for ids, mask, labels in loader:
            ids = ids.to(device)
            mask = mask.to(device)

            cnn_logits, bert_logits = model(ids, mask)

            actual.extend(labels.tolist())
            cnn_predictions.extend(cnn_logits.argmax(-1).cpu().tolist())
            bert_predictions.extend(bert_logits.argmax(-1).cpu().tolist())

    def metrics(predictions):
        return {
            "accuracy": float(accuracy_score(actual, predictions)),
            "macro_f1": float(f1_score(
                actual, predictions, average="macro"
            )),
            "confusion_matrix": confusion_matrix(
                actual, predictions, labels=[0, 1]
            ).tolist(),
        }

    return {
        "bert_original": metrics(bert_predictions),
        "bert_cnn": metrics(cnn_predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.epochs <= 10:
        raise ValueError("Epochs must be between 1 and 10.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not (BERT_PATH / "config.json").exists():
        raise FileNotFoundError(f"BERT checkpoint missing: {BERT_PATH}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    tag = (
        f"train{args.train_samples}_"
        f"val{args.validation_samples}_"
        f"epochs{args.epochs}"
    )

    report_path = RESULTS / f"bert_cnn_{tag}.json"
    head_path = CHECKPOINTS / f"bert_cnn_{tag}.pt"

    if report_path.exists():
        raise RuntimeError("Results already exist. Existing run preserved.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("BERT:", BERT_PATH)
    print("Train samples:", args.train_samples)
    print("Validation samples:", args.validation_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        str(BERT_PATH), local_files_only=True
    )

    pretrained = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_PATH), local_files_only=True
    )

    model = BertCNN(pretrained).to(device)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("CNN trainable parameters:", trainable)

    train_data = Reviews("train", args.train_samples, SEED)
    val_data = Reviews("validation", args.validation_samples, SEED + 1)

    train_loader = make_loader(train_data, tokenizer, True)
    val_loader = make_loader(val_data, tokenizer, False)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=2e-4,
        weight_decay=0.01,
    )

    best_f1 = -1.0
    best_epoch = None
    best_metrics = None
    history = []

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for ids, mask, labels in train_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(ids, mask)
            loss = F.cross_entropy(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=1.0,
            )

            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        average_loss = total_loss / len(train_loader)

        entry = {
            "epoch": epoch,
            "training_loss": average_loss,
            **metrics,
        }

        history.append(entry)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"Loss={average_loss:.4f} "
            f"BERT={metrics['bert_original']['accuracy']:.4f} "
            f"BERT+CNN={metrics['bert_cnn']['accuracy']:.4f} "
            f"F1={metrics['bert_cnn']['macro_f1']:.4f}"
        )

        current_f1 = metrics["bert_cnn"]["macro_f1"]

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_metrics = metrics

            # Save only the CNN head, not another copy of BERT.
            torch.save({
                "convs": model.convs.state_dict(),
                "classifier": model.classifier.state_dict(),
                "bert_source": str(BERT_PATH.relative_to(ROOT)),
                "max_length": MAX_LENGTH,
            }, head_path)

    report = {
        "project": "09",
        "architecture": "Frozen BERT + CNN",
        "dataset": "IMDB",
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "epochs": args.epochs,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "cnn_checkpoint": str(head_path.relative_to(ROOT)),
    }

    report_path.write_text(
        json.dumps(report, indent=4), encoding="utf-8"
    )

    print("\n===== PROJECT 09 BERT CNN RESULTS =====")
    print(json.dumps(report, indent=4))
    print("\nBERT + CNN training completed successfully!")


if __name__ == "__main__":
    main()