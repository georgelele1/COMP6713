from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from baseline.constants import DEFAULT_SPLIT_DIR
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. If omitted, auto-generates a parameter-named folder.",
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def build_parameter_output_name(args: argparse.Namespace) -> str:
    model_part = safe_name(args.model_name)
    lr_part = f"lr{args.learning_rate:g}"
    bs_part = f"bs{args.batch_size}"
    ep_part = f"ep{args.epochs}"
    ml_part = f"len{args.max_length}"
    wd_part = f"wd{args.weight_decay:g}"
    seed_part = f"seed{args.seed}"

    parts = [
        model_part,
        lr_part,
        bs_part,
        ep_part,
        ml_part,
        wd_part,
        seed_part,
    ]

    if args.max_train_samples is not None:
        parts.append(f"train{args.max_train_samples}")
    if args.max_valid_samples is not None:
        parts.append(f"valid{args.max_valid_samples}")
    if args.max_test_samples is not None:
        parts.append(f"test{args.max_test_samples}")

    return "_".join(parts)


def plot_metric(history_df: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    train_col = f"train_{metric_name}"
    valid_col = f"valid_{metric_name}"

    has_any = False

    if train_col in history_df.columns:
        plt.plot(history_df["epoch"], history_df[train_col], marker="o", label=train_col)
        has_any = True

    if valid_col in history_df.columns:
        plt.plot(history_df["epoch"], history_df[valid_col], marker="o", label=valid_col)
        has_any = True

    if not has_any:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{metric_name.replace('_', ' ').title()} over Epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_name}_curve.png", dpi=200)
    plt.close()


def plot_learning_rate(history_df: pd.DataFrame, output_dir: Path) -> None:
    if "learning_rate" not in history_df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["learning_rate"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_curve.png", dpi=200)
    plt.close()


def plot_training_history(history_path: Path, output_dir: Path) -> None:
    history_df = pd.read_csv(history_path)

    metrics_to_plot = [
        "loss",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    for metric in metrics_to_plot:
        plot_metric(history_df, metric, output_dir)

    plot_learning_rate(history_df, output_dir)


def main() -> None:
    args = parse_args()

    print("[1/8] Loading split datasets.")
    train_df = load_split_csv(args.train_path, args.max_train_samples)
    valid_df = load_split_csv(args.valid_path, args.max_valid_samples)
    test_df = load_split_csv(args.test_path, args.max_test_samples)

    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_name = build_parameter_output_name(args)
        output_dir = Path("outputs") / "bert_finetune" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "best_checkpoint"

    print(f"Output directory: {output_dir}")

    print("[2/8] Initializing fine-tuning model.")
    trainer = BertFineTuner(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        dropout=args.dropout,
    )

    print(f"Model: {args.model_name}")
    print(f"Device: {trainer.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Seed: {args.seed}")

    print("[3/8] Starting fine-tuning.")
    train_summary = trainer.train(
        train_df=train_df,
        valid_df=valid_df,
        epochs=args.epochs,
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience,
    )

    history_path = Path(train_summary["history_path"])

    print("[4/8] Plotting training curves.")
    if history_path.exists():
        plot_training_history(history_path, output_dir)
        print(f"Training curves saved in: {output_dir}")
    else:
        print(f"Warning: history file not found at {history_path}")

    print("[5/8] Reloading best checkpoint.")
    best_model = BertFineTuner.load(
        model_dir=checkpoint_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        seed=args.seed,
    )

    print("[6/8] Running test prediction.")
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

    print("[7/8] Computing metrics and exporting artifacts.")
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
                f"batch_size={args.batch_size}",
                f"epochs={args.epochs}",
                f"max_length={args.max_length}",
                f"learning_rate={args.learning_rate}",
                f"weight_decay={args.weight_decay}",
                f"seed={args.seed}",
            ]
        ),
        encoding="utf-8",
    )

    print("[8/8] Finished.")
    print(f"Best checkpoint: {checkpoint_dir}")
    print(f"Predictions saved to: {predictions_path}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()