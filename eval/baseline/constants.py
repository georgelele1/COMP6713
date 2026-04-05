"""项目级常量配置。

这个模块集中维护：
1. 项目根目录、数据目录、输出目录等路径；
2. 默认模型权重所在位置；
3. 情感分类任务使用的标签定义与显示顺序。

这样可以避免在不同脚本里重复写硬编码路径，也能保证
BERT、T5、评测脚本使用完全一致的标签顺序。
"""

from pathlib import Path

# 项目根目录：当前文件位于 baseline/ 下，所以向上取两级即可到项目根。
ROOT_DIR = Path(__file__).resolve().parents[1]

# 数据集目录。
DATASET_DIR = ROOT_DIR / "dataset"

# 合并清洗后的完整数据集文件。
DEFAULT_FULL_DATASET = DATASET_DIR / "merged_sentiment_clean.csv"

# 固定切分后的 train / valid / test 所在目录。
DEFAULT_SPLIT_DIR = DATASET_DIR / "splits"

# 默认测试集路径。
DEFAULT_TEST_DATASET = DEFAULT_SPLIT_DIR / "test.csv"

# 模型推理结果、指标、图表的统一输出目录。
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"

# BERT baseline 的本地模型目录。
DEFAULT_BERT_MODEL_DIR = (
    ROOT_DIR / "bert" / "nlptown_bert-base-multilingual-uncased-sentiment"
)

# FLAN-T5 baseline 的本地模型目录。
DEFAULT_T5_MODEL_DIR = ROOT_DIR / "T5" / "google_flan-t5-base"

# 项目统一标签定义：
# 0 = negative
# 1 = positive
# 2 = neutral
LABEL_ID_TO_NAME = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

# 反向映射：给定标签文本时，快速得到对应整数 id。
LABEL_NAME_TO_ID = {value: key for key, value in LABEL_ID_TO_NAME.items()}

# 所有评测、混淆矩阵、分类报告都使用同一顺序，避免结果错位。
LABEL_ORDER = [0, 1, 2]

# 给画图与报告导出使用的标签显示名称列表。
LABEL_DISPLAY_NAMES = [LABEL_ID_TO_NAME[label_id] for label_id in LABEL_ORDER]
