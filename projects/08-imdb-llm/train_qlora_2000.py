import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = ROOT / "checkpoints/project08_imdb/qwen_qlora_2000"
RESULT = ROOT / "results/project08_imdb/qlora_2000.json"

SEED = 42
MAX_LENGTH = 256


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer):
        self.items = []

        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end_id is None:
            raise ValueError("Missing Qwen end token.")

        for row in records:
            review_ids = tokenizer.encode(
                row["text"],
                add_special_tokens=False
            )[:160]

            review = tokenizer.decode(
                review_ids,
                skip_special_tokens=True
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
                    "content": "Review:\n" + review + "\nSentiment:"
                }
            ]

            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            answer = (
                "positive" if int(row["label"]) == 1
                else "negative"
            )

            answer_ids = tokenizer.encode(
                answer,
                add_special_tokens=False
            ) + [end_id]

            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            if len(ids) > MAX_LENGTH:
                raise ValueError(
                    f"Sequence exceeds {MAX_LENGTH} tokens."
                )

            padding = MAX_LENGTH - len(ids)

            self.items.append({
                "input_ids": (
                    ids + [tokenizer.pad_token_id] * padding
                ),
                "attention_mask": (
                    [1] * len(ids) + [0] * padding
                ),
                "labels": labels + [-100] * padding
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    return {
        key: torch.tensor(
            [item[key] for item in batch],
            dtype=torch.long
        )
        for key in ("input_ids", "attention_mask", "labels")
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if ADAPTER.exists():
        raise RuntimeError(
            "Smoke-test adapter already exists. "
            "Existing checkpoint preserved."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_ID)
    print("Compute dtype:", dtype)

    train = load_from_disk(
        str(ROOT / "data/processed/train")
    )

    labels = np.asarray(train["label"], dtype=np.int64)

    rng = np.random.default_rng(SEED)

    negative = rng.choice(
        np.where(labels == 0)[0],
        size=1000,
        replace=False
    )

    positive = rng.choice(
        np.where(labels == 1)[0],
        size=1000,
        replace=False
    )

    indices = rng.permutation(
        np.concatenate([negative, positive])
    )

    records = [train[int(i)] for i in indices]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )

    print("\nLoading Qwen in 4-bit...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
        quantization_config=quantization,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora)

    print("\n===== TRAINABLE PARAMETERS =====")
    model.print_trainable_parameters()

    dataset = SentimentDataset(records, tokenizer)

    output_dir = ROOT / "results/project08_imdb/qlora_2000_temp"

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=250,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="adamw_torch",
        logging_steps=2,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate
    )

    print("\n===== PROJECT 08 QLORA TRAINING =====")

    output = trainer.train()

    if not np.isfinite(output.training_loss):
        raise ValueError("Training loss is not finite.")

    ADAPTER.parent.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))

    result = {
        "model": MODEL_ID,
        "dataset": "IMDB",
        "split": "train",
        "method": "QLoRA 4-bit NF4",
        "samples_in_training_pool": 2000,
        "optimizer_steps": 250,
        "gradient_accumulation": 8,
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "training_loss": float(output.training_loss),
        "adapter": str(ADAPTER.relative_to(ROOT)),
        "validation_accuracy": None,
        "test_accuracy": None
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)

    RESULT.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print("\n===== PROJECT 08 QLORA 2000 RESULTS =====")
    print(json.dumps(result, indent=4))
    print("\nQLoRA 2000 training completed successfully!")


if __name__ == "__main__":
    main()