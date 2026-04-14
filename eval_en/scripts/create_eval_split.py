"""Create fixed train / valid / test splits.

Even if your current task mainly focuses on baseline inference, the project
still usually needs a fixed test set so that baseline and finetuned results
can be compared fairly.

This script does the following:
1. Read the full merged dataset.
2. Perform stratified sampling by label.
3. Create deterministic train / valid / test splits with a fixed random seed.
4. Save the split files to dataset/splits/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# When this file is run directly with `python scripts\create_eval_split.py`,
# Python only adds the scripts directory to sys.path by default.
# We manually prepend the project root so the baseline package can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.constants import DEFAULT_FULL_DATASET, DEFAULT_SPLIT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the split script."""
    parser = argparse.ArgumentParser(
        description="Create a deterministic stratified train/valid/test split from the merged dataset."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_FULL_DATASET),
        help="Merged dataset CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SPLIT_DIR),
        help="Directory where train.csv, valid.csv, and test.csv will be written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.92)
    parser.add_argument("--valid-ratio", type=float, default=0.04)
    parser.add_argument("--test-ratio", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    """Main execution flow."""
    args = parse_args()

    # Validate that the three split ratios sum to 1.
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train-ratio + valid-ratio + test-ratio must equal 1.0")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    # Create the output directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the full dataset.
    df = pd.read_csv(input_path)
    required_columns = {"id", "text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Input dataset must contain columns: {sorted(required_columns)}"
        )

    # First split out the training set, leaving a temporary valid+test pool.
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        stratify=df["label"],
        random_state=args.seed,
    )

    # Then split the temporary pool into validation and test sets.
    valid_share_in_temp = args.valid_ratio / (args.valid_ratio + args.test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_share_in_temp,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    # Sort by id before saving so the output files are stable and easier to inspect.
    train_df = train_df.sort_values("id").reset_index(drop=True)
    valid_df = valid_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    # Save the three split files.
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(output_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    # Print split sizes so the result can be checked quickly in the terminal.
    print(f"Saved split files to: {output_dir}")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test: {len(test_df)}")


if __name__ == "__main__":
    main()
