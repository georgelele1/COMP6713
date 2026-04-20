# COMP6713 — Sentiment Analysis Project

A sentiment analysis system that classifies text into **negative**, **positive**, and **neutral** categories. The project benchmarks two pre-trained baselines (BERT and FLAN-T5) against a fine-tuned BERT model, using a merged dataset drawn from two public sources.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Label Convention](#label-convention)
- [Dataset](#dataset)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Running the Models](#running-the-models)
  - [1. Create Dataset Splits](#1-create-dataset-splits)
  - [2. BERT Baseline (Pre-trained)](#2-bert-baseline-pre-trained)
  - [3. FLAN-T5 Baseline (Pre-trained)](#3-flan-t5-baseline-pre-trained)
  - [4. Fine-tuned BERT](#4-fine-tuned-bert)
- [Demo](#demo)
- [Outputs](#outputs)
- [Model Downloads](#model-downloads)
- [Configuration Reference](#configuration-reference)

---

## Project Overview

This project explores three approaches to three-class sentiment classification:

| Approach | Model | Method |
|---|---|---|
| Baseline (rule-based) | NLTK Opinion Lexicon | Word counting |
| Pre-trained baseline | `nlptown/bert-base-multilingual-uncased-sentiment` | Direct inference, star-rating remapping |
| Generative baseline | `google/flan-t5-base` | Prompt-based text generation |
| Fine-tuned | `bert-base-uncased` | Fine-tuned on merged dataset |

All models are evaluated on the same fixed test set using Accuracy, Macro Precision, Macro Recall, and Macro F1.

---

## Project Structure

```
COMP6713/
├── Dataset.py                  # Builds and saves the merged dataset CSV
├── dataset_analysis.py         # Exploratory data analysis
├── baseline.ipynb              # Notebook for baseline experiments
├── Sentiment_Word_Counting.py  # Rule-based sentiment counting
├── TF-IDF_Baseline.py          # TF-IDF baseline
├── outputs/                    # Top-level output directory
│
├── demo/
│   ├── demo.py                 # Gradio demo app (compares all models)
│   └── Readme_demo.md
│
└── eval/
    ├── baseline/               # Core inference package
    │   ├── __init__.py
    │   ├── constants.py        # Paths, label mappings, config constants
    │   ├── data_utils.py       # Dataset loading and output dir helpers
    │   ├── evaluation.py       # Metrics, confusion matrices, error analysis
    │   └── models/
    │       ├── bert_baseline.py      # BERT pre-trained inference wrapper
    │       ├── bert_finetune.py      # BERT fine-tuning trainer
    │       └── flan_t5_baseline.py   # FLAN-T5 generative inference wrapper
    ├── dataset/                # Stores merged CSV and train/valid/test splits
    │   └── splits/
    │       ├── train.csv
    │       ├── valid.csv
    │       └── test.csv
    ├── outputs/                # Per-run evaluation artifacts
    ├── scripts/
    │   └── create_eval_split.py  # Creates stratified train/valid/test splits
    ├── model_downloader.py     # Downloads model weights from HuggingFace
    ├── run_bert_baseline.py    # Entry point: BERT baseline inference
    ├── run_flan_t5_baseline.py # Entry point: FLAN-T5 baseline inference
    ├── run_finetune_bert.py    # Entry point: BERT fine-tuning
    ├── requirements.txt
    └── run.sh                  # Full setup and run script
```

---

## Label Convention

All models and dataset files use the same label mapping throughout the project:

| ID | Sentiment |
|----|-----------|
| 0  | negative  |
| 1  | positive  |
| 2  | neutral   |

---

## Dataset

The dataset is built by merging two public HuggingFace datasets:

- **`ma2za/many_emotions`** — Multi-emotion classification dataset. Original labels (anger, fear, joy, love, sadness, surprise, neutral) are remapped to the three-class scheme.
- **`mteb/tweet_sentiment_extraction`** — Tweet-level sentiment dataset. Labels are remapped to match the project convention.

After merging, the pipeline deduplicates by text content and saves a clean CSV.

**To rebuild the dataset from scratch:**

```bash
# From the project root
python Dataset.py
# Output: merged_sentiment_clean.csv
```

The merged CSV must be placed at `eval/dataset/merged_sentiment_clean.csv` before running the split script.

---

## Setup

### Requirements

- Python 3.10
- PyTorch (install separately before `requirements.txt`)
- Conda (recommended)

### Installation

```bash
# 1. Create and activate a conda environment
conda create -n comment-sa python=3.10 -y
conda activate comment-sa

# 2. Install PyTorch (adjust for your CUDA version if needed)
pip install torch torchvision torchaudio

# 3. Install remaining dependencies
pip install -r eval/requirements.txt
```

**Verify your environment:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import transformers, pandas, sklearn, sentencepiece, seaborn; print('OK')"
```

### Dependencies (`requirements.txt`)

```
accelerate==1.4.0
huggingface-hub==0.36.0
matplotlib==3.10.1
numpy==1.26.4
pandas==2.2.3
safetensors==0.5.3
scikit-learn==1.6.1
seaborn==0.13.2
sentencepiece==0.2.0
tqdm==4.67.1
transformers==4.49.0
```

---

## Quickstart

Run the full pipeline end-to-end from inside the `eval/` directory:

```bash
cd eval

# Step 1 – create fixed train/valid/test splits
python scripts/create_eval_split.py

# Step 2 – run BERT baseline on the test set
python run_bert_baseline.py --data-mode test

# Step 3 – run FLAN-T5 baseline on the test set
python run_flan_t5_baseline.py --data-mode test
```

---

## Running the Models

All run scripts must be executed from inside the `eval/` directory.

### 1. Create Dataset Splits

Creates stratified train (92%) / valid (4%) / test (4%) splits from the merged dataset. Splits are deterministic via a fixed random seed.

```bash
python scripts/create_eval_split.py
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--input-path` | `dataset/merged_sentiment_clean.csv` | Path to the merged dataset |
| `--output-dir` | `dataset/splits/` | Where to save the split files |
| `--seed` | `42` | Random seed for reproducibility |
| `--train-ratio` | `0.92` | Proportion for training |
| `--valid-ratio` | `0.04` | Proportion for validation |
| `--test-ratio` | `0.04` | Proportion for testing |

---

### 2. BERT Baseline (Pre-trained)

Uses `nlptown/bert-base-multilingual-uncased-sentiment` directly without any fine-tuning. The model's original 5-star output is remapped to three classes:

- 1–2 stars → negative
- 3 stars → neutral
- 4–5 stars → positive

```bash
# Run on fixed test split (recommended)
python run_bert_baseline.py --data-mode test

# Run on the full dataset
python run_bert_baseline.py --data-mode full

# Quick debug run (1000 samples)
python run_bert_baseline.py --data-mode test --max-samples 1000
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--data-mode` | `test` | `test` or `full` |
| `--model-dir` | `bert/nlptown_bert-base-multilingual-uncased-sentiment` | Local BERT checkpoint directory |
| `--batch-size` | `32` | Inference batch size |
| `--max-length` | `256` | Tokenizer max length |
| `--max-samples` | `None` | Cap samples (for debugging) |
| `--output-dir` | `outputs/bert_baseline/<mode>/` | Where to save results |
| `--device` | auto-detect | e.g. `cuda`, `cuda:0`, `cpu` |

---

### 3. FLAN-T5 Baseline (Pre-trained)

Uses `google/flan-t5-base` with a fixed prompt template. The model generates one of `negative`, `positive`, or `neutral` as free text, which is then parsed back to a label ID.

**Prompt template used:**
```
Classify the sentiment of the following comment as negative, positive, or neutral.

Comment: "{text}"

Answer with only one word: negative, positive, or neutral.
```

```bash
# Run on fixed test split
python run_flan_t5_baseline.py --data-mode test

# Quick debug run (200 samples)
python run_flan_t5_baseline.py --data-mode test --max-samples 200
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--data-mode` | `test` | `test` or `full` |
| `--model-dir` | `T5/google_flan-t5-base` | Local FLAN-T5 checkpoint directory |
| `--batch-size` | `32` | Inference batch size |
| `--max-input-length` | `256` | Max tokenized input length |
| `--max-new-tokens` | `4` | Max tokens to generate per prediction |
| `--max-samples` | `None` | Cap samples (for debugging) |
| `--output-dir` | `outputs/flan_t5_baseline/<mode>/` | Where to save results |
| `--device` | auto-detect | e.g. `cuda`, `cuda:0`, `cpu` |

---

### 4. Fine-tuned BERT

Fine-tunes `bert-base-uncased` on the merged training split with early stopping based on validation Macro F1.

```bash
python run_finetune_bert.py \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--train-path` | `dataset/splits/train.csv` | Training split |
| `--valid-path` | `dataset/splits/valid.csv` | Validation split |
| `--test-path` | `dataset/splits/test.csv` | Test split |
| `--model-name` | `bert-base-uncased` | HuggingFace model name |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `16` | Training batch size |
| `--max-length` | `256` | Tokenizer max length |
| `--learning-rate` | `2e-5` | AdamW learning rate |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--early-stopping-patience` | `2` | Epochs without improvement before stopping |
| `--scheduler-type` | `linear` | LR scheduler: `linear` or `cosine` |
| `--warmup-ratio` | `0.1` | Proportion of steps used for warmup |
| `--seed` | `42` | Random seed |
| `--output-dir` | auto-generated | Custom output path |
| `--device` | auto-detect | e.g. `cuda`, `cuda:0`, `cpu` |

---

## Demo

The demo app uses [Gradio](https://gradio.app) to compare all models side-by-side in a browser interface.

```bash
cd demo
python demo.py
```

Then open `http://127.0.0.1:7860` in your browser.

**Options:**

```bash
python demo.py --host 0.0.0.0 --port 7860 --share
```

The demo shows predictions from:
- **Baseline** — NLTK opinion lexicon (rule-based)
- **Pre-trained BERT** — requires model files at `demo/models/bert/`
- **Pre-trained FLAN-T5** — requires model files at `demo/models/T5/`
- **Fine-tuned BERT** — requires model files at `demo/models/finetuned_bert/`
- **Fine-tuned XLM-R** — requires model files at `demo/models/finetuned_xlmr/`

If a model directory is missing, the demo shows `missing` for that row rather than crashing.

---

## Outputs

Each run saves the following artifacts to its output directory:

| File | Description |
|---|---|
| `predictions.csv` | Per-sample predictions with gold label, predicted label, confidence, and raw model output |
| `metrics.json` | Overall metrics: accuracy, macro/weighted precision, recall, F1 |
| `per_class_metrics.csv` | Per-class precision, recall, F1, and support |
| `classification_report.csv` | Full sklearn classification report |
| `confusion_matrix_raw.png` | Raw count confusion matrix heatmap |
| `confusion_matrix_normalized.png` | Row-normalized confusion matrix heatmap |
| `misclassified_examples.csv` | All incorrectly predicted samples for error analysis |

Fine-tuning additionally saves:

| File | Description |
|---|---|
| `training_history.csv` | Per-epoch train/valid loss and metrics |
| `best_checkpoint/` | Best model weights (by validation Macro F1) |
| `train_summary.txt` | Hyperparameter and best-epoch summary |
| `*_curve.png` | Training curves (loss, accuracy, F1, learning rate) |

---

## Model Downloads

Pre-trained model weights must be downloaded locally before running inference. Use the provided downloader or download manually from HuggingFace.

**FLAN-T5:**
```bash
cd eval
python model_downloader.py
# Downloads to: T5/google_flan-t5-base/
```

**BERT (nlptown):**

Download manually from [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) and place the files at:
```
eval/bert/nlptown_bert-base-multilingual-uncased-sentiment/
```

Both directories must contain a `config.json` file for the scripts to detect them correctly.

---

## Configuration Reference

All paths and label constants are centralised in `eval/baseline/constants.py`. Edit this file if your directory layout differs from the defaults.

| Constant | Default Path |
|---|---|
| `DEFAULT_FULL_DATASET` | `dataset/merged_sentiment_clean.csv` |
| `DEFAULT_SPLIT_DIR` | `dataset/splits/` |
| `DEFAULT_TEST_DATASET` | `dataset/splits/test.csv` |
| `DEFAULT_OUTPUT_DIR` | `outputs/` |
| `DEFAULT_BERT_MODEL_DIR` | `bert/nlptown_bert-base-multilingual-uncased-sentiment/` |
| `DEFAULT_T5_MODEL_DIR` | `T5/google_flan-t5-base/` |
