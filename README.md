# Custom BPE Tokenizer

Phase 1 trains a custom BPE tokenizer from CoEdit text and Enron email bodies.
Phase 2 trains a small T5-style model from scratch for professional email rewriting.

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

## Prepare Phase 2 Data

```bash
python model/prepare_dataset.py
```

This creates:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

## Train The T5-Style Model

```bash
python model/train_model.py
```

This trains a small encoder-decoder model from scratch and saves it to:

```text
model/saved_model/
```

## Run Inference

```bash
python model/inference.py "sir i cant submit today can u give me more time"
```

## Evaluate The Model

```bash
python model/evaluate_model.py
```

This saves:

```text
model/evaluation_results.json
```

## Run The App

```bash
streamlit run app/streamlit_app.py
```
