from __future__ import annotations

import argparse
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
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration


MODEL_DIR = ROOT / "model" / "saved_model"
INSTRUCTION = "Rewrite this email professionally and concisely. Do not use em dashes."
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 256


def format_prompt(text: str) -> str:
    return f"{INSTRUCTION}\n\nInput: {text}"


def post_process(text: str) -> str:
    text = text.replace("—", ",").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?,])\1+", r"\1", text)
    return text.strip()


def load_model(model_dir: Path = MODEL_DIR):
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_rewrite(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    model: T5ForConditionalGeneration,
    device: torch.device,
) -> str:
    inputs = tokenizer(
        format_prompt(text),
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=MAX_OUTPUT_LENGTH,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return post_process(output)


def rewrite_email(text: str, model_dir: Path = MODEL_DIR) -> str:
    tokenizer, model, device = load_model(model_dir)
    return generate_rewrite(text, tokenizer, model, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a rough email professionally.")
    parser.add_argument("text", nargs="+", help="Rough email text")
    args = parser.parse_args()

    print(rewrite_email(" ".join(args.text)))


if __name__ == "__main__":
    main()
