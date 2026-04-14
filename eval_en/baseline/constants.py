"""Project-level constant configuration.

This module centralizes:
1. Project root, dataset, and output paths.
2. Default local model directories.
3. Label definitions and display order used throughout the task.

Keeping these values in one place avoids hard-coded paths across files and
ensures that BERT, T5, and evaluation scripts all use the same label order.
"""

from pathlib import Path

# Project root directory: this file is inside baseline/, so moving up one level
# reaches the project root.
ROOT_DIR = Path(__file__).resolve().parents[1]

# Dataset directory.
DATASET_DIR = ROOT_DIR / "dataset"

# Fully merged and cleaned dataset file.
DEFAULT_FULL_DATASET = DATASET_DIR / "merged_sentiment_clean.csv"

# Directory containing the fixed train / valid / test splits.
DEFAULT_SPLIT_DIR = DATASET_DIR / "splits"

# Default test split path.
DEFAULT_TEST_DATASET = DEFAULT_SPLIT_DIR / "test.csv"

# Unified output directory for predictions, metrics, and plots.
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"

# Local model directory for the BERT baseline.
DEFAULT_BERT_MODEL_DIR = (
    ROOT_DIR / "bert" / "nlptown_bert-base-multilingual-uncased-sentiment"
)

# Local model directory for the FLAN-T5 baseline.
DEFAULT_T5_MODEL_DIR = ROOT_DIR / "T5" / "google_flan-t5-base"

# Unified label definition used across the project:
# 0 = negative
# 1 = positive
# 2 = neutral
LABEL_ID_TO_NAME = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

# Reverse mapping from label text to integer id.
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}

# Shared label order used for evaluation, confusion matrices, and reports.
LABEL_ORDER = [0, 1, 2]

# Human-readable label names for plots and exported reports.
LABEL_DISPLAY_NAMES = [LABEL_ID_TO_NAME[label_id] for label_id in LABEL_ORDER]
