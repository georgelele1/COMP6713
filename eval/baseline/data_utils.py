"""数据读取与输出目录工具函数。

这里不做模型推理，只负责：
1. 根据 full / test 模式解析输入数据路径；
2. 读取 CSV 并做最基础的数据清洗；
3. 创建本次运行的输出目录。

通过把这些逻辑抽离成公共函数，可以让 BERT 和 T5 两套脚本
共享完全相同的数据入口，减少重复代码。
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from baseline.constants import DEFAULT_FULL_DATASET, DEFAULT_TEST_DATASET


# 项目要求输入数据至少具备这三列，后续评测和保存预测结果都依赖它们。
REQUIRED_COLUMNS = ["id", "text", "label"]

# 合法标签集合。
ALLOWED_LABELS = {0, 1, 2}


def resolve_input_path(data_mode: str, input_path: Optional[str]) -> Path:
    """根据运行模式返回实际要读取的 CSV 路径。"""
    if input_path:
        return Path(input_path)
    if data_mode == "test":
        return DEFAULT_TEST_DATASET
    return DEFAULT_FULL_DATASET


def load_dataset_frame(
    data_mode: str,
    input_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> tuple[pd.DataFrame, Path]:
    """读取数据集并返回 DataFrame 与对应路径。"""
    dataset_path = resolve_input_path(data_mode, input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "If you want test mode, run scripts/create_eval_split.py first "
            "or pass --input-path explicitly."
        )

    # 读取 CSV 文件。
    df = pd.read_csv(dataset_path)

    # 检查关键字段是否齐全。
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset file {dataset_path} is missing columns: {missing_columns}"
        )

    # 只保留本项目真正需要的字段，避免额外列干扰。
    df = df[REQUIRED_COLUMNS].copy()

    # 统一字段类型，确保后续 tokenizer 和指标计算稳定。
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # 删除清洗后为空的文本样本。
    df = df[df["text"] != ""].reset_index(drop=True)

    # 校验标签是否落在项目约定的三分类集合中。
    # 注意：这里的 neutral 标签 id 是 2；
    # BERT 原始 5 星输出里的“3-star”只是中性星级，不是最终数据集标签 3。
    found_labels = set(df["label"].unique().tolist())
    unexpected_labels = sorted(found_labels - ALLOWED_LABELS)
    if unexpected_labels:
        raise ValueError(
            "Unexpected label ids found in dataset: "
            f"{unexpected_labels}. Expected labels are 0=negative, 1=positive, 2=neutral."
        )

    # 调试模式下可以只取前若干条样本，加速冒烟测试。
    if max_samples is not None:
        df = df.head(max_samples).copy()

    return df, dataset_path


def build_output_dir(base_dir: Optional[str], model_key: str, data_mode: str) -> Path:
    """创建输出目录。"""
    if base_dir:
        output_dir = Path(base_dir)
    else:
        output_dir = Path("outputs") / model_key / data_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
