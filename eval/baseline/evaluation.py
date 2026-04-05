"""统一评测与结果导出模块。

本模块负责：
1. 根据预测结果计算总体指标与分类别指标；
2. 导出 metrics.json、classification_report.csv 等表格；
3. 生成 raw / normalized confusion matrix；
4. 单独保存错分样本，便于后续做 error analysis。

这样做的目的，是让 BERT 和 FLAN-T5 共用同一套评测逻辑，
确保结果口径一致，方便你和队友直接横向对比。
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
    """计算总体指标与分类别指标。"""
    # 转成列表后可直接传给 sklearn 的指标函数。
    y_true = predictions_df["gold_label"].tolist()
    y_pred = predictions_df["pred_label"].tolist()

    # 宏平均：每一类权重相同，更适合当前这种类别分布不完全均衡的数据。
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    # 加权平均：会考虑每一类样本量，适合作为补充指标。
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    # average=None 表示单独返回每个类别的指标。
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average=None,
            zero_division=0,
        )
    )

    # 汇总总体指标。accuracy 直观，macro_f1 更适合作为报告重点指标。
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

    # 逐类整理成表格，方便直接导出为 CSV。
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
    """保存一次推理运行的全部评测产物。"""
    # 确保输出目录存在。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 先计算数值指标，再统一导出。
    metrics, per_class_df = compute_metrics(predictions_df)
    metrics["model_name"] = model_name

    metrics_path = output_dir / "metrics.json"
    per_class_path = output_dir / "per_class_metrics.csv"
    report_path = output_dir / "classification_report.csv"

    # 以 JSON 形式保存总体指标，便于后续脚本读取。
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 分类别指标保存为表格，便于直接贴进报告。
    per_class_df.to_csv(per_class_path, index=False, encoding="utf-8-sig")

    # classification_report 会同时给出每类指标、macro avg、weighted avg 等信息。
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

    # 额外保存两种混淆矩阵图。
    save_confusion_matrix_plots(predictions_df, output_dir, model_name)

    # 单独保存错分样本，方便后续人工抽样分析错误类型。
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
    """保存原始计数版与归一化版混淆矩阵。"""
    y_true = predictions_df["gold_label"].tolist()
    y_pred = predictions_df["pred_label"].tolist()

    # 原始计数版：能看出每个格子里到底有多少条样本。
    raw_cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    # 归一化版：更适合看“某一真实类别被分错到哪里”。
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
    """绘制并保存热力图形式的混淆矩阵。"""
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
