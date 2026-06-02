from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = ROOT / ".tmp"
TEMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(TEMP_DIR))

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)


TOKENIZER_DIR = ROOT / "tokenizer" / "hf_tokenizer"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.jsonl"
BASE_MODEL_DIR = ROOT / "model" / "base_model"
OUTPUT_DIR = ROOT / "model" / "saved_model"

MAX_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 5
RANDOM_SEED = 42
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))


class EmailRewriteDataset(Dataset):
    def __init__(self, path: Path, tokenizer: PreTrainedTokenizerFast):
        self.examples = load_jsonl(path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt = format_prompt(example["instruction"], example["input"])
        output = clean_output(example["output"], self.tokenizer.eos_token or "")
        full_text = f"{prompt} {output}"

        tokenized = self.tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length

        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]

        tokenized.pop("token_type_ids", None)
        tokenized["labels"] = labels
        return tokenized


@dataclass
class CausalRewriteCollator:
    tokenizer: PreTrainedTokenizerFast
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if max_length % self.pad_to_multiple_of:
            max_length += self.pad_to_multiple_of - max_length % self.pad_to_multiple_of

        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_prompt(instruction: str, input_text: str) -> str:
    return f"Instruction: {instruction}\nInput: {input_text}\nOutput:"


def clean_output(output: str, eos_token: str) -> str:
    output = output.replace("—", ",").replace("–", "-").strip()
    if eos_token and not output.endswith(eos_token):
        output = f"{output} {eos_token}"
    return output


def load_tokenizer() -> PreTrainedTokenizerFast:
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {TOKENIZER_DIR}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id.")
    tokenizer.model_max_length = MAX_LENGTH
    return tokenizer


def load_base_model(tokenizer: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Missing base model directory: {BASE_MODEL_DIR}\n"
            "Run model/pretrain_base_model.py first to train the custom GPT-style "
            "base model from scratch with your tokenizer."
        )

    model = GPT2LMHeadModel.from_pretrained(str(BASE_MODEL_DIR))
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            "Base model vocabulary size does not match the current tokenizer. "
            "If you retrained the tokenizer, rerun model/pretrain_base_model.py."
        )
    model.config.use_cache = False
    return model


def precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def build_training_args() -> TrainingArguments:
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in supported_args else "evaluation_strategy"
    )
    fp16, bf16 = precision_flags()
    args = {
        "output_dir": str(OUTPUT_DIR),
        "overwrite_output_dir": True,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        strategy_name: "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "fp16": fp16,
        "bf16": bf16,
        "report_to": [],
        "save_total_limit": 2,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": RANDOM_SEED,
        "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    }
    args = {key: value for key, value in args.items() if key in supported_args}
    return TrainingArguments(**args)


def tokenizer_trainer_arg(tokenizer: PreTrainedTokenizerFast) -> dict[str, PreTrainedTokenizerFast]:
    supported_args = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in supported_args:
        return {"processing_class": tokenizer}
    if "tokenizer" in supported_args:
        return {"tokenizer": tokenizer}
    return {}


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_base_model(tokenizer)

    train_dataset = EmailRewriteDataset(TRAIN_PATH, tokenizer)
    validation_dataset = EmailRewriteDataset(VALIDATION_PATH, tokenizer)
    data_collator = CausalRewriteCollator(tokenizer=tokenizer)
    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        **tokenizer_trainer_arg(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved model to: {OUTPUT_DIR}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")


if __name__ == "__main__":
    main()
