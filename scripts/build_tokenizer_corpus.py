from __future__ import annotations

import csv
import html
import re
import sys
import unicodedata
from email import policy
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "processed" / "tokenizer_corpus.txt"

COEDIT_FILES = [
    ROOT / "data" / "raw" / "coedit" / "train.csv",
    ROOT / "data" / "raw" / "coedit" / "validation.csv",
    ROOT / "CoEdit" / "train.csv",
    ROOT / "CoEdit" / "validation.csv",
]
ENRON_FILES = [
    ROOT / "data" / "raw" / "enron" / "emails.csv",
    ROOT / "Enron" / "emails.csv",
]
MAX_ENRON_EMAILS = 100_000

HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")


def set_large_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def clean_text(text: str, min_chars: int = 20) -> str | None:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("—", ",").replace("–", "-")
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\ufffd", "")
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if len(text) < min_chars:
        return None

    return text


def existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []

    for path in paths:
        resolved = path.resolve()
        if path.exists() and resolved not in seen:
            seen.add(resolved)
            result.append(path)

    return result


def iter_coedit_text(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row.get("src", "")
            yield row.get("tgt", "")


def extract_email_body(raw_message: str) -> str:
    try:
        message = Parser(policy=policy.default).parsestr(raw_message)
        body = message.get_body(preferencelist=("plain", "html"))
        if body is not None:
            return body.get_content()
    except Exception:
        pass

    parts = raw_message.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else raw_message


def iter_enron_text(path: Path, max_emails: int = MAX_ENRON_EMAILS):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if index >= max_emails:
                break
            yield extract_email_body(row.get("message", ""))


def write_clean_lines(output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "coedit_files": 0,
        "enron_files": 0,
        "raw_samples": 0,
        "written_lines": 0,
        "duplicates": 0,
        "discarded": 0,
    }
    seen: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output:
        for path in existing_paths(COEDIT_FILES):
            counts["coedit_files"] += 1
            for text in iter_coedit_text(path):
                counts["raw_samples"] += 1
                cleaned = clean_text(text)
                if cleaned is None:
                    counts["discarded"] += 1
                    continue
                if cleaned in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(cleaned)
                output.write(cleaned + "\n")
                counts["written_lines"] += 1

        for path in existing_paths(ENRON_FILES):
            counts["enron_files"] += 1
            for text in iter_enron_text(path):
                counts["raw_samples"] += 1
                cleaned = clean_text(text)
                if cleaned is None:
                    counts["discarded"] += 1
                    continue
                if cleaned in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(cleaned)
                output.write(cleaned + "\n")
                counts["written_lines"] += 1

    return counts


def main() -> None:
    set_large_csv_limit()
    counts = write_clean_lines(OUTPUT_PATH)

    print(f"Wrote corpus to: {OUTPUT_PATH}")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
