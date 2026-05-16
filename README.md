# Custom BPE Tokenizer

Phase 1 trains a custom BPE tokenizer from CoEdit text and Enron email bodies.

## Setup

```bash
source /home/ali-musavi/jupyter_env/bin/activate
pip install -r requirements.txt
```

## Build The Corpus

```bash
python scripts/build_tokenizer_corpus.py
```

This creates:

```text
data/processed/tokenizer_corpus.txt
```

The corpus file is ignored by Git because it can be large.

## Train The Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

This creates:

```text
tokenizer/tokenizer.json
tokenizer/hf_tokenizer/
```

These tokenizer files are small enough to commit.

## Test The Tokenizer

```bash
python tokenizer/test_tokenizer.py
```

The test script prints tokenization examples, average tokens per sentence, and unknown-token rate.
