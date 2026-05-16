from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

COEDIT_TRAIN_PATHS = [
    ROOT / "data" / "raw" / "coedit" / "train.csv",
    ROOT / "CoEdit" / "train.csv",
]
COEDIT_VALIDATION_PATHS = [
    ROOT / "data" / "raw" / "coedit" / "validation.csv",
    ROOT / "CoEdit" / "validation.csv",
]

INSTRUCTION = "Rewrite this email professionally and concisely. Do not use em dashes."
PROMPT_RE = re.compile(
    r"^(?:Remove all grammatical errors from this text:|Improve the grammaticality:|"
    r"Fix grammaticality:|Paraphrase this sentence:|Rewrite this sentence:|"
    r"Simplify this sentence:|Make this sentence more formal:)\s*",
    re.IGNORECASE,
)

RANDOM_SEED = 42
MAX_COEDIT_TRAIN = 30_000
MAX_COEDIT_VALIDATION = 2_000


CUSTOM_EMAIL_EXAMPLES = [
    {
        "input": "sir i cant submit today can u give me more time",
        "output": "Dear Sir, I apologize, but I am unable to submit the work today. Could you please allow me some additional time?",
    },
    {
        "input": "hey can u send me the slides",
        "output": "Could you please send me the slides?",
    },
    {
        "input": "mam please check my assignment i uploaded it",
        "output": "Dear Madam, could you please review the assignment I uploaded?",
    },
    {
        "input": "i missed the meeting can you tell me what happened",
        "output": "I apologize for missing the meeting. Could you please share the main points discussed?",
    },
    {
        "input": "sorry for late reply i was busy",
        "output": "I apologize for the delayed response. I was occupied with other commitments.",
    },
    {
        "input": "can you give feedback on my resume",
        "output": "Could you please provide feedback on my resume when you have time?",
    },
    {
        "input": "i will come to the seminar tomorrow",
        "output": "I am writing to confirm that I will attend the seminar tomorrow.",
    },
    {
        "input": "i want internship in your company please tell me",
        "output": "I am interested in applying for an internship at your company. Could you please share any available opportunities?",
    },
    {
        "input": "send me the report fast",
        "output": "Could you please send me the report at your earliest convenience?",
    },
    {
        "input": "teacher i did not understand the topic can you explain again",
        "output": "Dear Teacher, I did not fully understand the topic. Could you please explain it again?",
    },
    {
        "input": "i need leave tomorrow because of family work",
        "output": "I would like to request leave tomorrow due to a family commitment.",
    },
    {
        "input": "please reply me today this is urgent",
        "output": "Could you please respond today? This matter is urgent.",
    },
    {
        "input": "can we move the meeting to monday",
        "output": "Could we please reschedule the meeting to Monday?",
    },
    {
        "input": "i attached the file check it",
        "output": "I have attached the file for your review.",
    },
    {
        "input": "i am waiting for your answer",
        "output": "I look forward to your response.",
    },
    {
        "input": "client said no what should i do now",
        "output": "The client declined the proposal. Could you please advise me on the next steps?",
    },
    {
        "input": "my internet was not working so i could not submit",
        "output": "I apologize for not submitting on time. My internet connection was unavailable.",
    },
    {
        "input": "tell me when you are free for call",
        "output": "Please let me know when you are available for a call.",
    },
    {
        "input": "i completed the task please check",
        "output": "I have completed the task. Could you please review it?",
    },
    {
        "input": "can you explain the project requirements again",
        "output": "Could you please explain the project requirements again?",
    },
]


def existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def clean_text(text: str) -> str:
    text = text.replace("—", ",").replace("–", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_source_instruction(text: str) -> str:
    return clean_text(PROMPT_RE.sub("", text))


def make_example(input_text: str, output_text: str) -> dict[str, str]:
    return {
        "instruction": INSTRUCTION,
        "input": clean_text(input_text),
        "output": clean_text(output_text),
    }


def read_coedit(path: Path, max_rows: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = strip_source_instruction(row.get("src", ""))
            target = clean_text(row.get("tgt", ""))
            if source and target:
                examples.append(make_example(source, target))
            if len(examples) >= max_rows:
                break

    return examples


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(RANDOM_SEED)

    train_path = existing_path(COEDIT_TRAIN_PATHS)
    validation_path = existing_path(COEDIT_VALIDATION_PATHS)
    if train_path is None or validation_path is None:
        raise FileNotFoundError("CoEdit train.csv and validation.csv are required.")

    train_examples = read_coedit(train_path, MAX_COEDIT_TRAIN)
    validation_examples = read_coedit(validation_path, MAX_COEDIT_VALIDATION)
    custom_examples = [make_example(item["input"], item["output"]) for item in CUSTOM_EMAIL_EXAMPLES]

    random.shuffle(custom_examples)
    test_examples = custom_examples[:6] + validation_examples[:44]
    validation_examples = custom_examples[6:12] + validation_examples[44:]
    train_examples = custom_examples[12:] + train_examples

    write_jsonl(PROCESSED_DIR / "train.jsonl", train_examples)
    write_jsonl(PROCESSED_DIR / "validation.jsonl", validation_examples)
    write_jsonl(PROCESSED_DIR / "test.jsonl", test_examples)

    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Wrote files to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
