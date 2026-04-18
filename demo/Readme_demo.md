# Sentiment Classification Demo

This project provides a demo to compare:
- Baseline (rule-based)
- Pre-trained models (BERT, FLAN-T5)
- Fine-tuned models (BERT, XLM-RoBERTa)

---

## 1. Requirements

- Python 3.10+
- torch
- transformers
- gradio
- nltk

Install dependencies:

pip install torch transformers gradio nltk

---

## 2. Model Setup

Due to size limits, models are NOT included in this repository.

You need to download and place them in the following structure:

models/
 ├── bert/
 ├── T5/
 ├── finetuned_bert/
 └── finetuned_xlmr/

---

### 2.1 Pre-trained Models (HuggingFace)

Download using:

# BERT
huggingface-cli download nlptown/bert-base-multilingual-uncased-sentiment --local-dir models/bert/nlptown_bert-base-multilingual-uncased-sentiment

# FLAN-T5
huggingface-cli download google/flan-t5-base --local-dir models/T5/google_flan-t5-base

---

### 2.2 Fine-tuned Models

Download from Google Drive:

- Fine-tuned BERT:
[PUT YOUR LINK HERE]

- Fine-tuned XLM-RoBERTa:
[PUT YOUR LINK HERE]

After downloading, place them as:

models/finetuned_bert/
models/finetuned_xlmr/

---

## 3. Run the Demo

python demo.py

Then open:

http://127.0.0.1:7860

---

## 4. Notes

- If port 7860 is occupied, change port:

python demo.py --port 7861

- The demo compares predictions across all models interactively.