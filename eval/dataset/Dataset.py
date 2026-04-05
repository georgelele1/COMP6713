"""构建合并情感数据集的脚本。

本脚本负责把两个不同来源的公开数据集统一到同一个三分类标签体系：
- 0 = negative
- 1 = positive
- 2 = neutral

主要步骤：
1. 读取 many_emotions 数据集并重映射标签；
2. 读取 tweet_sentiment_extraction 数据集并重映射标签；
3. 转为 pandas 后合并；
4. 做基础清洗与去重；
5. 生成统一 id；
6. 导出 merged_sentiment_clean.csv。
"""

from datasets import load_dataset, concatenate_datasets
import pandas as pd

# =====================================================
# 1. Load many_emotions
# This dataset currently uses an old HF dataset script, so we
# read the raw JSONL split files directly.
# target label:
# 0 = negative
# 1 = positive
# 2 = neutral
# =====================================================
ds1 = load_dataset(
    "json",
    data_files={
        "train": "https://huggingface.co/datasets/ma2za/many_emotions/resolve/main/data/split_dataset_train.jsonl.gz",
        "validation": "https://huggingface.co/datasets/ma2za/many_emotions/resolve/main/data/split_dataset_validation.jsonl.gz",
        "test": "https://huggingface.co/datasets/ma2za/many_emotions/resolve/main/data/split_dataset_test.jsonl.gz",
    },
)

many_emotions_map = {
    0: 0,  # anger -> negative
    1: 0,  # fear -> negative
    2: 1,  # joy -> positive
    3: 1,  # love -> positive
    4: 0,  # sadness -> negative
    5: 2,  # surprise -> neutral
    6: 2,  # neutral -> neutral
}


def remap_many_emotions(example):
    # 将 many_emotions 的原始标签映射为项目统一的三分类标签。
    return {
        "text": example["text"],
        "label": int(many_emotions_map[int(example["label"])]),
    }


# 把 train / validation / test 三个 split 合并成一个完整数据集。
all_ds1 = concatenate_datasets([
    ds1["train"],
    ds1["validation"],
    ds1["test"],
])

# 使用 map 批量重构字段，只保留 text 与 label 两列。
all_ds1 = all_ds1.map(
    remap_many_emotions,
    remove_columns=all_ds1.column_names,
)

print("many_emotions columns:", all_ds1.column_names)
print("many_emotions rows:", len(all_ds1))


# =====================================================
# 2. Load tweet_sentiment_extraction
# screenshot shows:
# 0 = negative
# 1 = neutral
# 2 = positive
# we need:
# 0 = negative
# 1 = positive
# 2 = neutral
# =====================================================
ds2 = load_dataset("mteb/tweet_sentiment_extraction")

tweet_map = {
    0: 0,  # negative -> negative
    1: 2,  # neutral -> neutral
    2: 1,  # positive -> positive
}


def remap_tweet_sentiment(example):
    # 将 tweet_sentiment_extraction 的标签顺序调整为项目统一格式。
    return {
        "text": example["text"],
        "label": int(tweet_map[int(example["label"])]),
    }


# 把该数据集的所有 split 合并。
all_splits_ds2 = [ds2[split] for split in ds2.keys()]
all_ds2 = concatenate_datasets(all_splits_ds2)

# 同样只保留 text 与 label 两列。
all_ds2 = all_ds2.map(
    remap_tweet_sentiment,
    remove_columns=all_ds2.column_names,
)

print("tweet_sentiment columns:", all_ds2.column_names)
print("tweet_sentiment rows:", len(all_ds2))


# =====================================================
# 3. Convert both to pandas first, then merge
# =====================================================
# 转为 pandas 是为了后续更方便地做拼接、清洗和去重。
df1 = all_ds1.to_pandas()
df2 = all_ds2.to_pandas()

# force same types
# 强制统一字段类型，避免两个来源的数据在拼接后出现类型不一致。
df1["text"] = df1["text"].astype(str)
df1["label"] = df1["label"].astype(int)

df2["text"] = df2["text"].astype(str)
df2["label"] = df2["label"].astype(int)

# 纵向拼接两个数据集。
merged_df = pd.concat([df1, df2], ignore_index=True)

print("before dedup:", len(merged_df))


# =====================================================
# 4. Clean and remove duplicates
# =====================================================
# 删除 text 或 label 缺失的样本。
merged_df = merged_df.dropna(subset=["text", "label"])

# 去除文本首尾空格。
merged_df["text"] = merged_df["text"].str.strip()

# 删除空文本。
merged_df = merged_df[merged_df["text"] != ""]

# remove duplicate rows by text only
# 仅按文本去重，避免相同评论在合并后重复保留。
merged_df = merged_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("after dedup:", len(merged_df))


# =====================================================
# 5. Create ordered id and keep only 3 columns
# =====================================================
# 为最终数据集生成连续 id，便于后续切分与结果回写。
merged_df.insert(0, "id", range(len(merged_df)))

# 最终只保留项目真正需要的三列。
merged_df = merged_df[["id", "text", "label"]]

print(merged_df.head())


# =====================================================
# 6. Save
# =====================================================
# 导出为带 BOM 的 UTF-8，方便 Windows 下直接打开查看。
merged_df.to_csv("merged_sentiment_clean.csv", index=False, encoding="utf-8-sig")
print("Saved: merged_sentiment_clean.csv")
