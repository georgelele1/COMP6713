"""BERT baseline 运行入口脚本。

你平时直接在命令行运行这个文件即可，例如：
python run_bert_baseline.py --data-mode test

它负责把数据读取、模型推理、结果保存、指标计算这几步串起来。
"""

from __future__ import annotations

import argparse

import pandas as pd

from baseline.constants import DEFAULT_BERT_MODEL_DIR
from baseline.data_utils import build_output_dir, load_dataset_frame
from baseline.evaluation import save_metrics_artifacts
from baseline.models.bert_baseline import BertSentimentBaseline


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
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
    """主执行流程。"""
    args = parse_args()
    print("[1/5] Loading dataset...")

    # 读取数据集，可在 test / full 之间切换。
    df, dataset_path = load_dataset_frame(
        data_mode=args.data_mode,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples: {len(df)}")

    # 创建本次运行的输出目录。
    output_dir = build_output_dir(args.output_dir, "bert_baseline", args.data_mode)
    print(f"Output directory: {output_dir}")

    # 初始化本地 BERT baseline。
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

    # 对全部文本做批量推理。
    print("[3/5] Starting batch inference...")
    predictions = model.predict(df["text"].tolist())

    # 统一整理为结果表，便于后续保存和评测。
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

    # 保存逐样本预测结果。
    print("[4/5] Saving predictions...")
    prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 计算并导出指标、分类报告、混淆矩阵等产物。
    print("[5/5] Computing metrics and exporting plots...")
    metrics = save_metrics_artifacts(
        prediction_df,
        output_dir=output_dir,
        model_name="bert_baseline",
    )

    # 在终端打印最关键的汇总指标。
    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")
    print("Run completed.")


if __name__ == "__main__":
    main()
