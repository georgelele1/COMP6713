# eval_en Project Guide

## 1. Project Overview

This project is used for baseline inference and result evaluation for comment sentiment analysis. It mainly contains two baseline routes:

- BERT baseline: `nlptown/bert-base-multilingual-uncased-sentiment`
- FLAN-T5 baseline: `google/flan-t5-base`

The project supports two data modes:

- `test`: run inference on the fixed test set, which is used for a fair comparison with the finetuned model
- `full`: run inference on the full dataset for additional analysis

The unified label definition is:

- `0 = negative`
- `1 = positive`
- `2 = neutral`

## 2. Directory Structure and File Purposes

### 2.1 Files in the project root

- `README.md`
  This project guide.

- `requirements.txt`
  Python dependencies required by the project, excluding `torch / torchvision / torchaudio`, which are usually installed separately.

- `run_bert_baseline.py`
  Entry script for BERT baseline inference. It handles dataset loading, local BERT weight loading, batched inference, prediction saving, metric computation, and plot export.

- `run_flan_t5_baseline.py`
  Entry script for FLAN-T5 baseline inference. It handles dataset loading, local FLAN-T5 weight loading, batched generative classification, prediction saving, metric computation, and plot export.

- `model_downloader.py`
  Helper script for downloading and caching model weights from Hugging Face into local directories.

- `run.sh`
  A simple script containing commonly used command-line examples for this project.

### 2.2 `baseline/` directory

This directory stores the shared core code used by the baseline inference pipelines.

- `baseline/__init__.py`
  Package initializer.

- `baseline/constants.py`
  Project constant definitions, including paths, default model directories, and label order.

- `baseline/data_utils.py`
  Utilities for dataset loading and basic cleaning, including `full/test` switching, CSV validation, label validation, and output directory creation.

- `baseline/evaluation.py`
  Shared evaluation and artifact export module, including Accuracy / Precision / Recall / F1, classification report export, confusion matrix plotting, and misclassified sample export.

### 2.3 `baseline/models/` directory

This directory contains model-specific inference wrappers.

- `baseline/models/bert_baseline.py`
  BERT baseline inference logic. The model natively outputs `1_star ~ 5_star`, and the current logic first takes the argmax over the 5-star outputs and then maps it into the three-class label space:
  - `1_star, 2_star -> negative`
  - `3_star -> neutral`
  - `4_star, 5_star -> positive`

- `baseline/models/flan_t5_baseline.py`
  FLAN-T5 baseline inference logic. It prompts the model to generate `negative / positive / neutral`, then parses the generated text into label ids.

### 2.4 `dataset/` directory

This directory stores files related to dataset construction and splitting.

- `dataset/Dataset.py`
  Script for building `merged_sentiment_clean.csv`. It reads multiple public datasets, unifies the label space, applies basic cleaning and deduplication, and exports the merged dataset.

- `dataset/merged_sentiment_clean.csv`
  The final merged and cleaned sentiment dataset.

- `dataset/splits/train.csv`
  Training split.

- `dataset/splits/valid.csv`
  Validation split.

- `dataset/splits/test.csv`
  Test split used for final baseline comparison.

### 2.5 `scripts/` directory

- `scripts/create_eval_split.py`
  Creates deterministic stratified `train/valid/test` splits from `merged_sentiment_clean.csv` using a fixed random seed.

### 2.6 `bert/` and `T5/` directories

This English copy of the project does not include the local weight directories. If you want to run the project, place the downloaded model weights in the following locations:

- `bert/nlptown_bert-base-multilingual-uncased-sentiment/`
- `T5/google_flan-t5-base/`

These directories are usually kept local only and excluded from Git because the files are large.

### 2.7 `outputs/` directory

This directory stores inference results.

- `outputs/bert_baseline/test/`
  BERT inference outputs on the test split.

- `outputs/flan_t5_baseline/test/`
  FLAN-T5 inference outputs on the test split.

Each run usually produces:

- `predictions.csv`
  Prediction result for each sample
- `metrics.json`
  Overall metrics
- `per_class_metrics.csv`
  Per-class metrics
- `classification_report.csv`
  sklearn classification report
- `confusion_matrix_raw.png`
  Raw-count confusion matrix
- `confusion_matrix_normalized.png`
  Normalized confusion matrix
- `misclassified_examples.csv`
  Misclassified examples

There are also analysis documents already saved under `outputs/`.

## 3. Environment Setup Commands

This project is recommended to run on **Windows + conda**, with dependencies installed using `pip`.

### 3.1 Create and activate the conda environment

```powershell
conda create -n comment-sa python=3.10 -y
conda activate comment-sa
```

### 3.2 Enter the project directory

```powershell
Set-Location D:\PyCharmProjects\eval_en
```

### 3.3 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 3.4 Install PyTorch

Recommended direct installation with pip:

```powershell
pip install torch torchvision torchaudio
```

If you prefer the official CUDA wheel from PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3.5 Install the remaining project dependencies

```powershell
pip install -r requirements.txt
```

### 3.6 Verify the environment

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import transformers, pandas, sklearn, sentencepiece, seaborn; print(transformers.__version__)"
```

## 4. Data Preparation Commands

### 4.1 Regenerate the merged dataset if needed

```powershell
python dataset\Dataset.py
```

### 4.2 Create deterministic train / valid / test splits

```powershell
python scripts\create_eval_split.py
```

## 5. Inference Commands

### 5.1 BERT baseline

Run on the test split:

```powershell
python run_bert_baseline.py --data-mode test
```

Run on the full dataset:

```powershell
python run_bert_baseline.py --data-mode full
```

Quick smoke test:

```powershell
python run_bert_baseline.py --data-mode test --max-samples 1000
```

Set batch size:

```powershell
python run_bert_baseline.py --data-mode test --batch-size 32
```

Specify device:

```powershell
python run_bert_baseline.py --data-mode test --device cuda
python run_bert_baseline.py --data-mode test --device cpu
```

### 5.2 FLAN-T5 baseline

Run on the test split:

```powershell
python run_flan_t5_baseline.py --data-mode test
```

Run on the full dataset:

```powershell
python run_flan_t5_baseline.py --data-mode full
```

Quick smoke test:

```powershell
python run_flan_t5_baseline.py --data-mode test --max-samples 200
```

Adjust batch size:

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16
python run_flan_t5_baseline.py --data-mode test --batch-size 24
```

Adjust input length:

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16 --max-input-length 128
```

Specify device:

```powershell
python run_flan_t5_baseline.py --data-mode test --device cuda
python run_flan_t5_baseline.py --data-mode test --device cpu
```

## 6. How to Read the Outputs

### 6.1 View overall metrics

```powershell
Get-Content outputs\bert_baseline\test\metrics.json
Get-Content outputs\flan_t5_baseline\test\metrics.json
```

### 6.2 View per-class metrics

```powershell
Get-Content outputs\bert_baseline\test\per_class_metrics.csv
Get-Content outputs\flan_t5_baseline\test\per_class_metrics.csv
```

### 6.3 View misclassified samples

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/bert_baseline/test/misclassified_examples.csv'); print(df.head().to_string())"
python -c "import pandas as pd; df=pd.read_csv('outputs/flan_t5_baseline/test/misclassified_examples.csv'); print(df.head().to_string())"
```

### 6.4 View prediction distribution

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs/bert_baseline/test/predictions.csv'); print(df['pred_text'].value_counts().to_string())"
python -c "import pandas as pd; df=pd.read_csv('outputs/flan_t5_baseline/test/predictions.csv'); print(df['pred_text'].value_counts().to_string())"
```

## 7. Important Notes for This Project

### 7.1 Label definition

The label definition must always remain:

- `0 = negative`
- `1 = positive`
- `2 = neutral`

Do not write `3 = neutral`.

### 7.2 Current BERT mapping logic

The current BERT baseline logic has been corrected to:

- first take the argmax among `1_star ~ 5_star`
- then map it into the three-class label space
- `1_star, 2_star -> negative`
- `3_star -> neutral`
- `4_star, 5_star -> positive`

This means that when `raw_prediction_text = 3_star`, the final `pred_text` must be `neutral`.

### 7.3 Main current issue for T5

The current FLAN-T5 results show that:

- the overall accuracy is not low
- but the recall for the `neutral` class is extremely low
- the model is clearly biased toward `negative` or `positive`

Therefore, when analyzing results, do not rely only on accuracy. Also inspect:

- `macro_f1`
- `per_class_metrics.csv`
- `confusion_matrix_normalized.png`

## 8. Common Issues

### 8.1 `ModuleNotFoundError: No module named 'baseline'`

If you run:

```powershell
python scripts\create_eval_split.py
```

and see this error, the script in this project already contains compatibility handling. Under normal conditions, rerunning it should work.

### 8.2 T5 inference is too slow

Try these first:

```powershell
python run_flan_t5_baseline.py --data-mode test --batch-size 16
python run_flan_t5_baseline.py --data-mode test --batch-size 16 --max-input-length 128
```

### 8.3 GPU is not detected

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If the output is `False`, the current PyTorch environment is not using CUDA correctly.

## 9. Recommended Execution Order

If you want to run the project from scratch, the recommended order is:

```powershell
conda create -n comment-sa python=3.10 -y
conda activate comment-sa
Set-Location D:\PyCharmProjects\eval_en
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
python scripts\create_eval_split.py
python run_bert_baseline.py --data-mode test --max-samples 1000
python run_flan_t5_baseline.py --data-mode test --max-samples 200
python run_bert_baseline.py --data-mode test
python run_flan_t5_baseline.py --data-mode test
```

## 10. Local Model Directory Note

The following directories are intended to stay local and should usually not be committed to Git:

- `bert/nlptown_bert-base-multilingual-uncased-sentiment/`
- `T5/google_flan-t5-base/`

Reasons:

- the files are large
- they are not suitable for version control

If the project uses a `.gitignore`, these directories should be ignored there.
