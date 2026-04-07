"""Fine-tuned BERT training + evaluation entry script."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline.constants import DEFAULT_SPLIT_DIR
from baseline.data_utils import build_output_dir
from baseline.evaluation import save_metrics_artifacts
from baseline.models.bert_finetune import BertFineTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune bert-base-uncased for 3-class sentiment classification."
    )
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_SPLIT_DIR / "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--valid-path",
        default=str(DEFAULT_SPLIT_DIR / "valid.csv"),
        help="Path to valid.csv",
    )
    parser.add_argument(
        "--test-path",
        default=str(DEFAULT_SPLIT_DIR / "test.csv"),
        help="Path to test.csv",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="HF model name for fine-tuning.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to outputs/bert_finetune/test/",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override such as cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def load_split_csv(path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"id", "text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset file {path} must contain columns: {sorted(required_columns)}")

    df = df[["id", "text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"] != ""].reset_index(drop=True)

    if max_samples is not None:
        df = df.head(max_samples).copy()

    return df


def main() -> None:
    args = parse_args()

    print("[1/7] Loading split datasets.")
    train_df = load_split_csv(args.train_path, args.max_train_samples)
    valid_df = load_split_csv(args.valid_path, args.max_valid_samples)
    test_df = load_split_csv(args.test_path, args.max_test_samples)

    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    output_dir = build_output_dir(args.output_dir, "bert_finetune", "test")
    checkpoint_dir = output_dir / "best_checkpoint"

    print(f"Output directory: {output_dir}")

    print("[2/7] Initializing fine-tuning model.")
    trainer = BertFineTuner(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    print(f"Model: {args.model_name}")
    print(f"Device: {trainer.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")

    print("[3/7] Starting fine-tuning.")
    train_summary = trainer.train(
        train_df=train_df,
        valid_df=valid_df,
        epochs=args.epochs,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
    )

    print("[4/7] Reloading best checkpoint.")
    best_model = BertFineTuner.load(
        model_dir=checkpoint_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    print("[5/7] Running test prediction.")
    predictions = best_model.predict(test_df["text"].tolist())

    prediction_df = pd.DataFrame(
        {
            "id": test_df["id"],
            "text": test_df["text"],
            "gold_label": test_df["label"],
            "pred_label": predictions.pred_label,
            "pred_text": predictions.pred_text,
            "confidence": predictions.confidence,
            "raw_prediction_text": predictions.raw_prediction_text,
            "model_name": "bert_finetune",
        }
    )

    predictions_path = output_dir / "predictions.csv"
    prediction_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("[6/7] Computing metrics and exporting artifacts.")
    metrics = save_metrics_artifacts(
        prediction_df,
        output_dir=output_dir,
        model_name="bert_finetune",
    )

    summary_path = output_dir / "train_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"model_name={args.model_name}",
                f"best_epoch={train_summary['best_epoch']}",
                f"best_valid_macro_f1={train_summary['best_valid_macro_f1']:.6f}",
                f"best_checkpoint_dir={train_summary['best_checkpoint_dir']}",
                f"train_history_path={train_summary['history_path']}",
            ]
        ),
        encoding="utf-8",
    )

    print("[7/7] Finished.")
    print(f"Best checkpoint: {checkpoint_dir}")
    print(f"Predictions saved to: {predictions_path}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()