"""FLAN-T5 baseline 运行入口脚本。

你可以像下面这样运行：
python run_flan_t5_baseline.py --data-mode test

这个脚本与 BERT 入口保持同样结构，方便两种模型的结果直接对齐比较。
"""

from __future__ import annotations

import argparse

import pandas as pd

from baseline.constants import DEFAULT_T5_MODEL_DIR
from baseline.data_utils import build_output_dir, load_dataset_frame
from baseline.evaluation import save_metrics_artifacts
from baseline.models.flan_t5_baseline import FlanT5SentimentBaseline


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
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
    """主执行流程。"""
    args = parse_args()
    print("[1/5] Loading dataset...")

    # 读取待评测数据，可在固定测试集和完整数据集之间切换。
    df, dataset_path = load_dataset_frame(
        data_mode=args.data_mode,
        input_path=args.input_path,
        max_samples=args.max_samples,
    )
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples: {len(df)}")

    # 创建本次运行输出目录。
    output_dir = build_output_dir(args.output_dir, "flan_t5_baseline", args.data_mode)
    print(f"Output directory: {output_dir}")

    # 初始化本地 FLAN-T5 模型。
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

    # 对文本逐批做生成式情感分类。
    print("[3/5] Starting batch inference...")
    predictions = model.predict(df["text"].tolist())

    # 统一整理为结果表，与 BERT 的输出字段尽量保持一致。
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

    # 保存逐样本预测结果，便于后续误差分析。
    print("[4/5] Saving predictions...")
    prediction_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 统一计算并导出指标、分类报告、混淆矩阵。
    print("[5/5] Computing metrics and exporting plots...")
    metrics = save_metrics_artifacts(
        prediction_df,
        output_dir=output_dir,
        model_name="flan_t5_baseline",
    )

    # 在终端打印关键汇总信息。
    print(f"Loaded dataset: {dataset_path}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")
    print("Metrics summary:")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Parse errors: {int(prediction_df['parse_error'].sum())}")
    print("Run completed.")


if __name__ == "__main__":
    main()
