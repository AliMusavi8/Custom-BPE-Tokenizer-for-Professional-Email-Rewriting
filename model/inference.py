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
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


MODEL_DIR = ROOT / "model" / "saved_model"
INSTRUCTION = "Rewrite this email professionally and concisely. Do not use em dashes."
MAX_INPUT_LENGTH = 160
MAX_NEW_TOKENS = 96
STOP_MARKERS = ("\nInstruction:", "\nInput:", "\nOutput:")


def format_prompt(text: str) -> str:
    return f"Instruction: {INSTRUCTION}\nInput: {text}\nOutput:"


def post_process(text: str) -> str:
    for marker in STOP_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    text = text.replace("—", ",").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([.!?,])\1+", r"\1", text)
    return text.strip()


def load_model(model_dir: Path = MODEL_DIR):
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define both pad_token_id and eos_token_id.")
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_rewrite(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    model: GPT2LMHeadModel,
    device: torch.device,
) -> str:
    inputs = tokenizer(
        format_prompt(text),
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_length:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
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
