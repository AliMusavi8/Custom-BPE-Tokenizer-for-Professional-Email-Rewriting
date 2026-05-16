from __future__ import annotations

import json
import inspect
import os
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
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
)


TOKENIZER_DIR = ROOT / "tokenizer" / "hf_tokenizer"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.jsonl"
OUTPUT_DIR = ROOT / "model" / "saved_model"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256


class EmailRewriteDataset(Dataset):
    def __init__(self, path: Path, tokenizer: PreTrainedTokenizerFast):
        self.examples = load_jsonl(path)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        prompt = format_prompt(example["instruction"], example["input"])

        model_input = self.tokenizer(
            prompt,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        model_input.pop("token_type_ids", None)
        labels = self.tokenizer(
            text_target=example["output"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
        model_input["labels"] = add_eos(labels["input_ids"], self.tokenizer.eos_token_id)
        return model_input


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def format_prompt(instruction: str, input_text: str) -> str:
    return f"{instruction}\n\nInput: {input_text}"


def add_eos(input_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None or input_ids[-1:] == [eos_token_id]:
        return input_ids
    if len(input_ids) >= MAX_TARGET_LENGTH:
        return input_ids[:-1] + [eos_token_id]
    return input_ids + [eos_token_id]


def load_tokenizer() -> PreTrainedTokenizerFast:
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {TOKENIZER_DIR}")

    return PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))


def build_model(tokenizer: PreTrainedTokenizerFast) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=256,
        d_ff=1024,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=4,
        dropout_rate=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    return T5ForConditionalGeneration(config)


def build_training_args(fp16: bool) -> Seq2SeqTrainingArguments:
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    args = {
        "output_dir": str(OUTPUT_DIR),
        "num_train_epochs": 5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "learning_rate": 5e-4,
        strategy_name: "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "fp16": fp16,
        "report_to": [],
        "save_total_limit": 2,
        "dataloader_pin_memory": fp16,
    }
    return Seq2SeqTrainingArguments(**args)


def main() -> None:
    tokenizer = load_tokenizer()
    model = build_model(tokenizer)

    train_dataset = EmailRewriteDataset(TRAIN_PATH, tokenizer)
    validation_dataset = EmailRewriteDataset(VALIDATION_PATH, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    fp16 = torch.cuda.is_available()
    training_args = build_training_args(fp16)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved model to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
