"""Build the merged sentiment dataset.

This script unifies two public datasets into a single three-class label space:
- 0 = negative
- 1 = positive
- 2 = neutral

Main steps:
1. Load the many_emotions dataset and remap its labels.
2. Load the tweet_sentiment_extraction dataset and remap its labels.
3. Convert both datasets to pandas and merge them.
4. Apply basic cleaning and deduplication.
5. Create a unified ordered id column.
6. Export merged_sentiment_clean.csv.
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
    # Map the original many_emotions labels into the unified project label space.
    return {
        "text": example["text"],
        "label": int(many_emotions_map[int(example["label"])]),
    }


# Merge the train / validation / test splits into a single full dataset.
all_ds1 = concatenate_datasets([
    ds1["train"],
    ds1["validation"],
    ds1["test"],
])

# Rebuild the dataset so that only text and label are kept.
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
    # Reorder tweet_sentiment_extraction labels into the project's label format.
    return {
        "text": example["text"],
        "label": int(tweet_map[int(example["label"])]),
    }


# Merge all splits of this dataset.
all_splits_ds2 = [ds2[split] for split in ds2.keys()]
all_ds2 = concatenate_datasets(all_splits_ds2)

# Again keep only text and label.
all_ds2 = all_ds2.map(
    remap_tweet_sentiment,
    remove_columns=all_ds2.column_names,
)

print("tweet_sentiment columns:", all_ds2.column_names)
print("tweet_sentiment rows:", len(all_ds2))


# =====================================================
# 3. Convert both to pandas first, then merge
# =====================================================
# Convert to pandas for easier concatenation, cleaning, and deduplication.
df1 = all_ds1.to_pandas()
df2 = all_ds2.to_pandas()

# Force consistent field types before concatenation.
df1["text"] = df1["text"].astype(str)
df1["label"] = df1["label"].astype(int)

df2["text"] = df2["text"].astype(str)
df2["label"] = df2["label"].astype(int)

# Concatenate the two datasets row-wise.
merged_df = pd.concat([df1, df2], ignore_index=True)

print("before dedup:", len(merged_df))


# =====================================================
# 4. Clean and remove duplicates
# =====================================================
# Drop samples whose text or label is missing.
merged_df = merged_df.dropna(subset=["text", "label"])

# Strip leading and trailing whitespace.
merged_df["text"] = merged_df["text"].str.strip()

# Remove empty texts.
merged_df = merged_df[merged_df["text"] != ""]

# Remove duplicate rows based only on text to avoid retaining repeated comments.
merged_df = merged_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("after dedup:", len(merged_df))


# =====================================================
# 5. Create ordered id and keep only 3 columns
# =====================================================
# Create a continuous id column for later splitting and result alignment.
merged_df.insert(0, "id", range(len(merged_df)))

# Keep only the three columns required by the project.
merged_df = merged_df[["id", "text", "label"]]

print(merged_df.head())


# =====================================================
# 6. Save
# =====================================================
# Export with UTF-8 BOM so the CSV opens cleanly on Windows.
merged_df.to_csv("merged_sentiment_clean.csv", index=False, encoding="utf-8-sig")
print("Saved: merged_sentiment_clean.csv")
