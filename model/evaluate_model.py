from __future__ import annotations

import json
from pathlib import Path

from inference import generate_rewrite, load_model


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
RESULTS_PATH = ROOT / "model" / "evaluation_results.json"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def word_overlap(reference: str, prediction: str) -> float:
    reference_words = set(reference.lower().split())
    prediction_words = set(prediction.lower().split())
    if not reference_words:
        return 0.0
    return len(reference_words & prediction_words) / len(reference_words)


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_word in left:
        current = [0]
        for index, right_word in enumerate(right, start=1):
            if left_word == right_word:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, prediction: str) -> float:
    reference_words = reference.lower().split()
    prediction_words = prediction.lower().split()
    if not reference_words or not prediction_words:
        return 0.0

    lcs = lcs_length(reference_words, prediction_words)
    precision = lcs / len(prediction_words)
    recall = lcs / len(reference_words)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def gleu_like(reference: str, prediction: str) -> float:
    reference_words = set(reference.lower().split())
    prediction_words = set(prediction.lower().split())
    if not reference_words or not prediction_words:
        return 0.0

    overlap = len(reference_words & prediction_words)
    precision = overlap / len(prediction_words)
    recall = overlap / len(reference_words)
    return min(precision, recall)


def sari_unigram(source: str, reference: str, prediction: str) -> float:
    source_words = set(source.lower().split())
    reference_words = set(reference.lower().split())
    prediction_words = set(prediction.lower().split())

    added = prediction_words - source_words
    should_add = reference_words - source_words
    kept = prediction_words & source_words
    should_keep = reference_words & source_words
    deleted = source_words - prediction_words
    should_delete = source_words - reference_words

    return (
        f1(added, should_add) + f1(kept, should_keep) + f1(deleted, should_delete)
    ) / 3


def f1(actual: set[str], expected: set[str]) -> float:
    if not actual and not expected:
        return 1.0
    if not actual or not expected:
        return 0.0

    overlap = len(actual & expected)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    examples = load_jsonl(TEST_PATH)
    rows = []
    tokenizer, model, device = load_model()

    for example in examples:
        prediction = generate_rewrite(example["input"], tokenizer, model, device)
        rows.append(
            {
                "input": example["input"],
                "reference": example["output"],
                "prediction": prediction,
                "word_overlap": word_overlap(example["output"], prediction),
                "rouge_l": rouge_l(example["output"], prediction),
                "gleu": gleu_like(example["output"], prediction),
                "sari": sari_unigram(
                    example["input"], example["output"], prediction
                ),
                "has_em_dash": "—" in prediction,
            }
        )

    em_dash_rate = sum(row["has_em_dash"] for row in rows) / len(rows) if rows else 0.0
    average_word_overlap = (
        sum(row["word_overlap"] for row in rows) / len(rows) if rows else 0.0
    )
    average_rouge_l = sum(row["rouge_l"] for row in rows) / len(rows) if rows else 0.0
    average_gleu = sum(row["gleu"] for row in rows) / len(rows) if rows else 0.0
    average_sari = sum(row["sari"] for row in rows) / len(rows) if rows else 0.0

    results = {
        "num_examples": len(rows),
        "em_dash_violation_rate": em_dash_rate,
        "average_word_overlap": average_word_overlap,
        "average_rouge_l": average_rouge_l,
        "average_gleu": average_gleu,
        "average_sari": average_sari,
        "examples": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Examples: {len(rows)}")
    print(f"Em dash violation rate: {em_dash_rate:.4%}")
    print(f"Average word overlap: {average_word_overlap:.4f}")
    print(f"Average ROUGE-L: {average_rouge_l:.4f}")
    print(f"Average GLEU: {average_gleu:.4f}")
    print(f"Average SARI: {average_sari:.4f}")
    print(f"Saved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
