"""Shared evaluation and artifact export module.

This module is responsible for:
1. Computing overall metrics and per-class metrics from predictions.
2. Exporting metrics.json, classification_report.csv, and related files.
3. Generating raw and normalized confusion matrices.
4. Saving misclassified samples for later error analysis.

The goal is to let BERT and FLAN-T5 share exactly the same evaluation logic,
so that their outputs are directly comparable.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from baseline.constants import LABEL_DISPLAY_NAMES, LABEL_ID_TO_NAME, LABEL_ORDER


def compute_metrics(predictions_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compute overall metrics and per-class metrics."""
    # Convert to lists so they can be passed directly to sklearn metric functions.
    y_true = predictions_df["gold_label"].tolist()
    y_pred = predictions_df["pred_label"].tolist()

    # Macro average: every class gets equal weight, which is useful for datasets
    # whose class distribution is not perfectly balanced.
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    # Weighted average: takes class frequency into account and serves as a
    # complementary summary metric.
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    # average=None returns a separate metric for each class.
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average=None,
            zero_division=0,
        )
    )

    # Summarize the main overall metrics.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "num_examples": int(len(predictions_df)),
    }

    # Organize per-class metrics as a table for CSV export.
    per_class_rows = []
    for index, label_id in enumerate(LABEL_ORDER):
        per_class_rows.append(
            {
                "label_id": label_id,
                "label_name": LABEL_ID_TO_NAME[label_id],
                "precision": float(per_class_precision[index]),
                "recall": float(per_class_recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
        )

    return metrics, pd.DataFrame(per_class_rows)


def save_metrics_artifacts(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> dict:
    """Save all evaluation artifacts for one inference run."""
    # Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics first, then export everything in a consistent way.
    metrics, per_class_df = compute_metrics(predictions_df)
    metrics["model_name"] = model_name

    metrics_path = output_dir / "metrics.json"
    per_class_path = output_dir / "per_class_metrics.csv"
    report_path = output_dir / "classification_report.csv"

    # Save overall metrics in JSON format for convenient downstream reuse.
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save per-class metrics as a table that can be pasted into reports.
    per_class_df.to_csv(per_class_path, index=False, encoding="utf-8-sig")

    # classification_report includes per-class metrics, macro avg, weighted avg, etc.
    report_dict = classification_report(
        predictions_df["gold_label"],
        predictions_df["pred_label"],
        labels=LABEL_ORDER,
        target_names=LABEL_DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        report_path,
        encoding="utf-8-sig",
    )

    # Save both raw-count and normalized confusion matrices.
    save_confusion_matrix_plots(predictions_df, output_dir, model_name)

    # Save misclassified examples separately for manual error analysis later.
    misclassified_df = predictions_df[
        predictions_df["gold_label"] != predictions_df["pred_label"]
    ].copy()
    misclassified_df.to_csv(
        output_dir / "misclassified_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


def save_confusion_matrix_plots(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
) -> None:
    """Save raw-count and normalized confusion matrix plots."""
    y_true = predictions_df["gold_label"].tolist()
    y_pred = predictions_df["pred_label"].tolist()

    # Raw-count version: shows the actual number of samples in each cell.
    raw_cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    # Normalized version: better for showing where each true class is misrouted.
    norm_cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="true")

    _plot_confusion_matrix(
        raw_cm,
        output_dir / "confusion_matrix_raw.png",
        f"{model_name} Confusion Matrix (Raw Count)",
        fmt="d",
    )
    _plot_confusion_matrix(
        norm_cm,
        output_dir / "confusion_matrix_normalized.png",
        f"{model_name} Confusion Matrix (Normalized)",
        fmt=".2f",
    )


def _plot_confusion_matrix(matrix, output_path: Path, title: str, fmt: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=LABEL_DISPLAY_NAMES,
        yticklabels=LABEL_DISPLAY_NAMES,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
