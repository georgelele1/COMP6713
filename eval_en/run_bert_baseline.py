"""BERT baseline entry script.

You can run this file directly from the command line, for example:
python run_bert_baseline.py --data-mode test

It connects dataset loading, model inference, prediction saving,
and metric export in a single workflow.
"""

from __future__ import annotations

import argparse

import pandas as pd

from baseline.constants import DEFAULT_BERT_MODEL_DIR
from baseline.data_utils import build_output_dir, load_dataset_frame
from baseline.evaluation import save_metrics_artifacts
from baseline.models.bert_baseline import BertSentimentBaseline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the local BERT sentiment baseline on a test split or the full dataset."
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
        default=str(DEFAULT_BERT_MODEL_DIR),
        help="Local directory containing the BERT checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/bert_baseline/<mode>/",
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

    # Load the evaluation dataset, switching between test and full modes.
    df, dataset_path = load_dataset_frame(
        data_mode=args.data_mode,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples: {len(df)}")

    # Create the output directory for this run.
    output_dir = build_output_dir(args.output_dir, "bert_baseline", args.data_mode)
    print(f"Output directory: {output_dir}")

    # Initialize the local BERT baseline model.
    print("[2/5] Loading BERT model...")
    model = BertSentimentBaseline(
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {model.device}")
    print(f"batch_size: {args.batch_size}, max_length: {args.max_length}")

    # Run batched inference over all texts.
    print("[3/5] Starting batch inference...")
    predictions = model.predict(df["text"].tolist())

    # Assemble a unified result table for saving and evaluation.
    prediction_df = pd.DataFrame(
        {
            "id": df["id"],
            "text": df["text"],
            "gold_label": df["label"],
            "pred_label": predictions.pred_label,
            "pred_text": predictions.pred_text,
            "confidence": predictions.confidence,
            "raw_prediction_text": predictions.raw_prediction_text,
            "model_name": "bert_baseline",
        }
    )

    # Save per-sample predictions.
    print("[4/5] Saving predictions...")
    prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # Compute and export metrics, reports, and confusion matrices.
    print("[5/5] Computing metrics and exporting plots...")
    metrics = save_metrics_artifacts(
        prediction_df,
        output_dir=output_dir,
        model_name="bert_baseline",
    )

    # Print the key summary metrics to the terminal.
    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")
    print("Run completed.")


if __name__ == "__main__":
    main()
