"""生成固定 train / valid / test 切分的脚本。

虽然你当前主要负责 baseline 推理，但为了和队友微调结果做公平比较，
项目通常仍然需要一份固定测试集。

这个脚本的作用是：
1. 从完整合并数据集中读取样本；
2. 按标签做分层抽样；
3. 用固定随机种子生成 train / valid / test；
4. 保存到 dataset/splits/ 目录下。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# 当使用 `python scripts\create_eval_split.py` 直接运行时，
# Python 默认只会把 scripts 目录加入模块搜索路径。
# 这里手动把项目根目录加入 sys.path，确保可以正常导入 baseline 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.constants import DEFAULT_FULL_DATASET, DEFAULT_SPLIT_DIR


def parse_args() -> argparse.Namespace:
    """解析切分脚本的命令行参数。"""
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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """主执行流程。"""
    args = parse_args()

    # 校验三部分比例之和是否等于 1。
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train-ratio + valid-ratio + test-ratio must equal 1.0")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    # 如果输出目录不存在则自动创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取完整数据集。
    df = pd.read_csv(input_path)
    required_columns = {"id", "text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Input dataset must contain columns: {sorted(required_columns)}"
        )

    # 第一步：先切出训练集，剩下部分作为 valid + test 的临时集合。
    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_ratio,
        stratify=df["label"],
        random_state=args.seed,
    )

    # 第二步：在剩余数据中继续按比例切分 valid 与 test。
    valid_share_in_temp = args.valid_ratio / (args.valid_ratio + args.test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_share_in_temp,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    # 为了让输出文件更稳定可读，这里按 id 排序后再保存。
    train_df = train_df.sort_values("id").reset_index(drop=True)
    valid_df = valid_df.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    # 保存三个切分文件。
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid_df.to_csv(output_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    # 在终端打印样本数量，便于快速确认切分结果。
    print(f"Saved split files to: {output_dir}")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test: {len(test_df)}")


if __name__ == "__main__":
    main()
