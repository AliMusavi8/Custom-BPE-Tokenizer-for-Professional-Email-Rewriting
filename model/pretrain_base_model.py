from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = ROOT / ".tmp"
TEMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(TEMP_DIR))

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


TOKENIZER_DIR = ROOT / "tokenizer" / "hf_tokenizer"
WIKITEXT_DIR = ROOT / "data" / "raw" / "wikitext-103-raw-v1"
OUTPUT_DIR = ROOT / "model" / "base_model"

BLOCK_SIZE = 256
MAX_TRAIN_LINES = 300_000
MAX_VALIDATION_LINES = 4_000
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
DATASET_NUM_PROC = int(os.environ.get("DATASET_NUM_PROC", "1"))
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "0"))

WHITESPACE_RE = re.compile(r"\s+")


def clean_wikitext_line(text: str) -> str | None:
    text = text.replace("—", ",").replace("–", "-")
    text = WHITESPACE_RE.sub(" ", text).strip()
    if len(text) < 40:
        return None
    if text.startswith("=") and text.endswith("="):
        return None
    return text


def load_tokenizer() -> PreTrainedTokenizerFast:
    if not TOKENIZER_DIR.exists():
        raise FileNotFoundError(f"Missing tokenizer directory: {TOKENIZER_DIR}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id.")
    tokenizer.model_max_length = BLOCK_SIZE
    return tokenizer


def select_rows(dataset: Dataset, max_rows: int) -> Dataset:
    if len(dataset) <= max_rows:
        return dataset
    return dataset.select(range(max_rows))


def prepare_lm_dataset(
    split: str,
    tokenizer: PreTrainedTokenizerFast,
    max_lines: int,
) -> Dataset:
    if not WIKITEXT_DIR.exists():
        raise FileNotFoundError(f"Missing WikiText dataset directory: {WIKITEXT_DIR}")

    dataset = select_rows(load_from_disk(str(WIKITEXT_DIR))[split], max_lines)
    num_proc = max(DATASET_NUM_PROC, 1)
    map_workers = {"num_proc": num_proc} if num_proc > 1 else {}

    def clean_batch(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned_texts = []
        for text in batch["text"]:
            cleaned = clean_wikitext_line(text)
            if cleaned is not None:
                cleaned_texts.append(cleaned)
        return {"text": cleaned_texts}

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=False, verbose=False)

    def group_texts(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        token_ids = []
        for input_ids in batch["input_ids"]:
            token_ids.extend(input_ids)
            if tokenizer.eos_token_id is not None:
                token_ids.append(tokenizer.eos_token_id)

        total_length = len(token_ids) // BLOCK_SIZE * BLOCK_SIZE
        token_ids = token_ids[:total_length]
        chunks = [
            token_ids[index : index + BLOCK_SIZE]
            for index in range(0, total_length, BLOCK_SIZE)
        ]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * BLOCK_SIZE for _ in chunks],
            "labels": [chunk.copy() for chunk in chunks],
        }

    cleaned = dataset.map(
        clean_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Cleaning WikiText {split}",
        **map_workers,
    )
    tokenized = cleaned.map(
        tokenize_batch,
        batched=True,
        remove_columns=cleaned.column_names,
        desc=f"Tokenizing WikiText {split}",
        **map_workers,
    )
    grouped = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1_000,
        remove_columns=tokenized.column_names,
        desc=f"Packing WikiText {split}",
        **map_workers,
    )

    if len(grouped) == 0:
        raise ValueError(f"No usable language-model chunks found for split: {split}")

    return grouped


def build_model(tokenizer: PreTrainedTokenizerFast) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_embd=256,
        n_layer=6,
        n_head=4,
        n_inner=1024,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
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
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        strategy_name: "steps",
        "eval_steps": 2_000,
        "save_steps": 2_000,
        "logging_steps": 100,
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
    model = build_model(tokenizer)

    train_dataset = prepare_lm_dataset("train", tokenizer, MAX_TRAIN_LINES)
    validation_dataset = prepare_lm_dataset("validation", tokenizer, MAX_VALIDATION_LINES)
    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        **tokenizer_trainer_arg(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Saved base model to: {OUTPUT_DIR}")
    print(f"Training chunks: {len(train_dataset)}")
    print(f"Validation chunks: {len(validation_dataset)}")


if __name__ == "__main__":
    main()
