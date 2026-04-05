# Baseline Inference Structure

## Directory layout

- `baseline/constants.py`
  Central paths, labels, and default model locations.
- `baseline/data_utils.py`
  Shared CSV loading logic and `full` / `test` mode handling.
- `baseline/evaluation.py`
  Metric calculation, classification report export, confusion matrix plots, and misclassified sample export.
- `baseline/models/bert_baseline.py`
  Wrapper for `nlptown/bert-base-multilingual-uncased-sentiment`.
- `baseline/models/flan_t5_baseline.py`
  Wrapper for `google/flan-t5-base`.
- `run_bert_baseline.py`
  CLI entry point for the BERT baseline.
- `run_flan_t5_baseline.py`
  CLI entry point for the FLAN-T5 baseline.
- `scripts/create_eval_split.py`
  Deterministic stratified split helper. Use it once to create `train/valid/test`.

## Typical workflow

1. Create a fixed split once:
   `python scripts/create_eval_split.py`
2. Run a model on the team test set:
   `python run_bert_baseline.py --data-mode test`
3. Optionally run on the full dataset:
   `python run_flan_t5_baseline.py --data-mode full`

## Output artifacts

Each run writes to `outputs/<model>/<mode>/` by default:

- `predictions.csv`
- `metrics.json`
- `per_class_metrics.csv`
- `classification_report.csv`
- `confusion_matrix_raw.png`
- `confusion_matrix_normalized.png`
- `misclassified_examples.csv`

