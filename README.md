# Custom BPE Tokenizer for Professional Email Rewriting

This project builds a small email rewriting system in three stages:

1. Train a custom BPE tokenizer on CoEdit text and Enron email language.
2. Train a small GPT-2-style decoder-only base model from scratch on WikiText using the custom tokenizer.
3. Continue training that custom model on rough-email to professional-email rewrite pairs.

The model does **not** load pretrained GPT-2, T5, or FLAN-T5 weights. It uses a GPT-2-style causal language model architecture, initialized from random weights, with the custom tokenizer trained in this project.

## How It Works

The tokenizer learns subword pieces using Byte Pair Encoding, so common words and email phrases can become compact tokens while rare words are split into smaller reusable parts. This helps the model handle professional email vocabulary without needing a huge vocabulary.

The rewriting model follows a prompt-completion pattern:

```text
Rough email + instruction
↓
Custom BPE tokenizer
↓
Custom GPT-2-style decoder-only model
↓
Professional rewritten email
```

## Dataset

This project uses two main datasets plus a small set of custom email examples:

```text
CoEdit
Enron Emails
WikiText-103
Custom email rewriting examples
```

### CoEdit

CoEdit is used for tokenizer training and rewrite-task training.

For tokenizer training, both the source and target text are treated as plain text:

```text
CoEdit source text
CoEdit target text
```

For rewrite-task training, CoEdit is the main paired dataset because it already contains input-output rewriting examples. The pairs are converted into this format:

```text
Instruction: Rewrite this email professionally and concisely. Do not use em dashes.
Input: rough email text
Output: professional rewritten email
```

### WikiText-103

WikiText-103 is used for base model training. It is stored locally as a Hugging Face Arrow dataset with a single `text` column. The base training script tokenizes the cleaned text and packs it into fixed-length causal language modeling blocks.

### Enron Emails

Enron is used only for tokenizer training. The email bodies are cleaned and added to the tokenizer corpus as plain text:

```text
Enron email bodies
```

This helps the tokenizer learn professional email language, greetings, closings, business vocabulary, names, dates, punctuation, and common email patterns.

Enron is not used as direct model training pairs because it does not naturally provide rough-email to professional-email examples.

### Custom Email Examples

Custom email rewriting examples are added to Phase 3 because the final task is professional email rewriting. These cover common situations such as requesting files, asking for deadline extensions, following up on meetings, apologizing for late submissions, confirming attendance, and requesting feedback.

## Project Structure

```text
scripts/build_tokenizer_corpus.py   Build cleaned tokenizer text corpus.
tokenizer/train_tokenizer.py        Train and save the custom BPE tokenizer.
tokenizer/test_tokenizer.py         Print tokenizer quality checks.

model/prepare_dataset.py            Create train/validation/test JSONL files.
model/pretrain_base_model.py        Train the custom base model from scratch.
model/train_model.py                Train the small GPT-style rewriting model.
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

## Phase 2: Base Model

Train the custom decoder-only base model from scratch on WikiText:

```bash
python model/pretrain_base_model.py
```

If local multiprocessing works on your machine, you can speed up dataset preprocessing and loading:

```bash
DATASET_NUM_PROC=4 DATALOADER_NUM_WORKERS=2 python model/pretrain_base_model.py
```

Output:

```text
model/base_model/
```

## Phase 3: Rewriting Model

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
