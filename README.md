# Custom BPE Tokenizer for Professional Email Rewriting

This project builds a small email rewriting system in two stages:

1. Train a custom BPE tokenizer on CoEdit text and Enron email language.
2. Train a small T5-style encoder-decoder model from scratch to rewrite rough emails into clearer, more professional emails.

The model is **not FLAN-T5** and it does **not** load pretrained T5 weights. It only uses the T5 architecture, initialized from random weights, with the custom tokenizer trained in this project.

## How It Works

The tokenizer learns subword pieces using Byte Pair Encoding, so common words and email phrases can become compact tokens while rare words are split into smaller reusable parts. This helps the model handle professional email vocabulary without needing a huge vocabulary.

The rewriting model follows an encoder-decoder pattern:

```text
Rough email + instruction
↓
Custom BPE tokenizer
↓
T5-style encoder-decoder model
↓
Professional rewritten email
```

## Dataset

This project uses two main datasets plus a small set of custom email examples:

```text
CoEdit
Enron Emails
Custom email rewriting examples
```

### CoEdit

CoEdit is used in both phases.

For tokenizer training, both the source and target text are treated as plain text:

```text
CoEdit source text
CoEdit target text
```

For model training, CoEdit is the main paired dataset because it already contains input-output rewriting examples. The pairs are converted into this format:


### Enron Emails

Enron is used only for tokenizer training. The email bodies are cleaned and added to the tokenizer corpus as plain text:

```text
Enron email bodies
```

This helps the tokenizer learn professional email language, greetings, closings, business vocabulary, names, dates, punctuation, and common email patterns.

Enron is not used as direct model training pairs because it does not naturally provide rough-email to professional-email examples.

### Custom Email Examples

Custom email rewriting examples are added to Phase 2 because the final task is professional email rewriting. These cover common situations such as requesting files, asking for deadline extensions, following up on meetings, apologizing for late submissions, confirming attendance, and requesting feedback.

## Project Structure

```text
scripts/build_tokenizer_corpus.py   Build cleaned tokenizer text corpus.
tokenizer/train_tokenizer.py        Train and save the custom BPE tokenizer.
tokenizer/test_tokenizer.py         Print tokenizer quality checks.

model/prepare_dataset.py            Create train/validation/test JSONL files.
model/train_model.py                Train the small T5-style rewriting model.
model/inference.py                  Rewrite one rough email using the trained model.
model/evaluate_model.py             Evaluate outputs and save metrics.

app/streamlit_app.py                Simple web app for email rewriting.
```

## Setup

```bash
source /home/ali-musavi/jupyter_env/bin/activate
pip install -r requirements.txt
```

## Phase 1: Tokenizer

Build the tokenizer corpus:

```bash
python scripts/build_tokenizer_corpus.py
```

Output:

```text
data/processed/tokenizer_corpus.txt
```

Train the tokenizer:

```bash
python tokenizer/train_tokenizer.py
```

Outputs:

```text
tokenizer/tokenizer.json
tokenizer/hf_tokenizer/
```

Test it:

```bash
python tokenizer/test_tokenizer.py
```

## Phase 2: Rewriting Model

Prepare paired training data:

```bash
python model/prepare_dataset.py
```

Outputs:

```text
data/processed/train.jsonl
data/processed/validation.jsonl
data/processed/test.jsonl
```

Train the model:

```bash
python model/train_model.py
```

Output:

```text
model/saved_model/
```

Run inference:

```bash
python model/inference.py "sir i cant submit today can u give me more time"
```

Evaluate:

```bash
python model/evaluate_model.py
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```