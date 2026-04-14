"""Utilities for dataset loading and output directories.

This module does not perform model inference. It is only responsible for:
1. Resolving the input dataset path based on full / test mode.
2. Reading CSV files and applying minimal cleaning.
3. Creating output directories for the current run.

Extracting this logic into shared functions allows the BERT and T5 scripts
to use the same data entry pipeline and reduces duplicated code.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from baseline.constants import DEFAULT_FULL_DATASET, DEFAULT_TEST_DATASET


# The project requires these columns for evaluation and prediction export.
REQUIRED_COLUMNS = ["id", "text", "label"]

# Allowed label ids.
ALLOWED_LABELS = {0, 1, 2}


def resolve_input_path(data_mode: str, input_path: Optional[str]) -> Path:
    """Return the actual CSV path to load for the selected mode."""
    if input_path:
        return Path(input_path)
    if data_mode == "test":
        return DEFAULT_TEST_DATASET
    return DEFAULT_FULL_DATASET


def load_dataset_frame(
    data_mode: str,
    input_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the dataset and return a DataFrame plus its resolved path."""
    dataset_path = resolve_input_path(data_mode, input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "If you want test mode, run scripts/create_eval_split.py first "
            "or pass --input-path explicitly."
        )

    # Read the CSV file.
    df = pd.read_csv(dataset_path)

    # Check that the required columns are present.
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset file {dataset_path} is missing columns: {missing_columns}"
        )

    # Keep only the fields that the project actually needs.
    df = df[REQUIRED_COLUMNS].copy()

    # Normalize data types so tokenization and metric computation remain stable.
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # Drop samples whose text becomes empty after cleaning.
    df = df[df["text"] != ""].reset_index(drop=True)

    # Validate that labels match the agreed three-class definition.
    # Note: the neutral label id in the dataset is 2;
    # the BERT model's original "3-star" output is only a neutral star rating,
    # not a final dataset label id of 3.
    found_labels = set(df["label"].unique().tolist())
    unexpected_labels = sorted(found_labels - ALLOWED_LABELS)
    if unexpected_labels:
        raise ValueError(
            "Unexpected label ids found in dataset: "
            f"{unexpected_labels}. Expected labels are 0=negative, 1=positive, 2=neutral."
        )

    # In debugging mode, optionally keep only the first N samples.
    if max_samples is not None:
        df = df.head(max_samples).copy()

    return df, dataset_path


def build_output_dir(base_dir: Optional[str], model_key: str, data_mode: str) -> Path:
    """Create and return the output directory for a run."""
    if base_dir:
        output_dir = Path(base_dir)
    else:
        output_dir = Path("outputs") / model_key / data_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
