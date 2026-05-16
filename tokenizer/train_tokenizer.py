from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "processed" / "tokenizer_corpus.txt"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
HF_TOKENIZER_DIR = ROOT / "tokenizer" / "hf_tokenizer"

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
VOCAB_SIZE = 8000
MIN_FREQUENCY = 2


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def train_tokenizer() -> Tokenizer:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\n"
            "Run scripts/build_tokenizer_corpus.py first."
        )

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train([str(CORPUS_PATH)], trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer) -> None:
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    HF_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(TOKENIZER_PATH))

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER_PATH),
        unk_token="<UNK>",
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        mask_token="<MASK>",
    )
    hf_tokenizer.save_pretrained(str(HF_TOKENIZER_DIR))


def main() -> None:
    tokenizer = train_tokenizer()
    save_tokenizer(tokenizer)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Corpus lines: {count_lines(CORPUS_PATH)}")
    print(f"Saved tokenizer: {TOKENIZER_PATH}")
    print(f"Saved Hugging Face tokenizer: {HF_TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
