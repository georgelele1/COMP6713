"""FLAN-T5 baseline entry script.

You can run it like this:
python run_flan_t5_baseline.py --data-mode test

This script follows the same structure as the BERT entry script so that
results from the two models can be compared directly.
"""

from __future__ import annotations

import argparse

import pandas as pd

from baseline.constants import DEFAULT_T5_MODEL_DIR
from baseline.data_utils import build_output_dir, load_dataset_frame
from baseline.evaluation import save_metrics_artifacts
from baseline.models.flan_t5_baseline import FlanT5SentimentBaseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the local FLAN-T5 sentiment baseline on a test split or the full dataset."
    )
    parser.add_argument(
        "--data-mode",
        choices=["test", "full"],
        default="test",
        help="Use the fixed test split or the full merged dataset.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional CSV path. Overrides the default path for the selected mode.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_T5_MODEL_DIR),
        help="Local directory containing the FLAN-T5 checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/flan_t5_baseline/<mode>/",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override such as cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution flow."""
    args = parse_args()
    print("[1/5] Loading dataset...")

    # Load evaluation data, switching between the fixed test set and the full dataset.
    df, dataset_path = load_dataset_frame(
        data_mode=args.data_mode,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples: {len(df)}")

    # Create the output directory for this run.
    output_dir = build_output_dir(args.output_dir, "flan_t5_baseline", args.data_mode)
    print(f"Output directory: {output_dir}")

    # Initialize the local FLAN-T5 model.
    print("[2/5] Loading FLAN-T5 model...")
    model = FlanT5SentimentBaseline(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {model.device}")
    print(
        f"batch_size: {args.batch_size}, "
        f"max_input_length: {args.max_input_length}, "
        f"max_new_tokens: {args.max_new_tokens}"
    )

    # Perform generative sentiment classification in batches.
    print("[3/5] Starting batch inference...")
    predictions = model.predict(df["text"].tolist())

    # Build a unified prediction table to align with the BERT output format.
    prediction_df = pd.DataFrame(
        {
            "id": df["id"],
            "text": df["text"],
            "gold_label": df["label"],
            "pred_label": predictions.pred_label,
            "pred_text": predictions.pred_text,
            "confidence": predictions.confidence,
            "raw_prediction_text": predictions.raw_prediction_text,
            "parse_error": predictions.parse_error,
            "model_name": "flan_t5_baseline",
        }
    )

    # Save per-sample predictions for later analysis.
    print("[4/5] Saving predictions...")
    prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # Compute and export metrics, reports, and confusion matrices.
    print("[5/5] Computing metrics and exporting plots...")
    metrics = save_metrics_artifacts(
        prediction_df,
        output_dir=output_dir,
        model_name="flan_t5_baseline",
    )

    # Print the most important run summary in the terminal.
    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")
    print("Run completed.")


if __name__ == "__main__":
    main()
