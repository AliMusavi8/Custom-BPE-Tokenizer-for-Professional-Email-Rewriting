from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"

SAMPLES = [
    "Could you please send me the assignment file by tomorrow?",
    "Dear Sir, I wanted to ask whether you reviewed my submission.",
    "Please let me know if the meeting is still scheduled.",
]


def load_tokenizer() -> PreTrainedTokenizerFast:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}\n"
            "Run tokenizer/train_tokenizer.py first."
        )

    return PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        unk_token="<UNK>",
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        mask_token="<MASK>",
    )


def main() -> None:
    tokenizer = load_tokenizer()
    total_tokens = 0
    unknown_tokens = 0

    print(f"Vocabulary size: {len(tokenizer)}")
    print()

    for sample in SAMPLES:
        tokens = tokenizer.tokenize(sample)
        ids = tokenizer.encode(sample, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        sample_unknowns = tokens.count("<UNK>")

        total_tokens += len(tokens)
        unknown_tokens += sample_unknowns

        print(f"Text: {sample}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Unknown tokens: {sample_unknowns}")
        print(f"Decoded: {decoded}")
        print()

    average_tokens = total_tokens / len(SAMPLES)
    unknown_rate = unknown_tokens / total_tokens if total_tokens else 0.0

    print(f"Average tokens per sentence: {average_tokens:.2f}")
    print(f"Unknown token rate: {unknown_rate:.4%}")


if __name__ == "__main__":
    main()
