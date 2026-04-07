from datasets import load_dataset, concatenate_datasets
import pandas as pd

# =====================================================
# Project unified label mapping
# 0 = negative
# 1 = positive
# 2 = neutral
# =====================================================

PROJECT_LABELS = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}

# =====================================================
# 1. Load many_emotions
# original:
# 0 anger
# 1 fear
# 2 joy
# 3 love
# 4 sadness
# 5 surprise
# 6 neutral
#
# project target:
# 0 negative
# 1 positive
# 2 neutral
# =====================================================
ds1 = load_dataset("ma2za/many_emotions", "split")

many_emotions_map = {
    0: PROJECT_LABELS["negative"],  # anger
    1: PROJECT_LABELS["negative"],  # fear
    2: PROJECT_LABELS["positive"],  # joy
    3: PROJECT_LABELS["positive"],  # love
    4: PROJECT_LABELS["negative"],  # sadness
    5: PROJECT_LABELS["neutral"],   # surprise
    6: PROJECT_LABELS["neutral"],   # neutral
}

def remap_many_emotions(example):
    return {
        "text": str(example["text"]),
        "label": int(many_emotions_map[int(example["label"])]),
    }

all_ds1 = concatenate_datasets([
    ds1["train"],
    ds1["validation"],
    ds1["test"],
])

all_ds1 = all_ds1.map(
    remap_many_emotions,
    remove_columns=all_ds1.column_names,
)

print("many_emotions columns:", all_ds1.column_names)
print("many_emotions rows:", len(all_ds1))
print("many_emotions label counts:")
print(pd.Series(all_ds1["label"]).value_counts().sort_index())


# =====================================================
# 2. Load tweet_sentiment_extraction
# original dataset labels:
# 0 = negative
# 1 = neutral
# 2 = positive
#
# project target:
# 0 = negative
# 1 = positive
# 2 = neutral
# =====================================================
ds2 = load_dataset("mteb/tweet_sentiment_extraction")

tweet_map = {
    0: PROJECT_LABELS["negative"],  # negative
    1: PROJECT_LABELS["neutral"],   # neutral
    2: PROJECT_LABELS["positive"],  # positive
}

def remap_tweet_sentiment(example):
    return {
        "text": str(example["text"]),
        "label": int(tweet_map[int(example["label"])]),
    }

all_splits_ds2 = [ds2[split] for split in ds2.keys()]
all_ds2 = concatenate_datasets(all_splits_ds2)

all_ds2 = all_ds2.map(
    remap_tweet_sentiment,
    remove_columns=all_ds2.column_names,
)

print("tweet_sentiment columns:", all_ds2.column_names)
print("tweet_sentiment rows:", len(all_ds2))
print("tweet_sentiment label counts:")
print(pd.Series(all_ds2["label"]).value_counts().sort_index())


# =====================================================
# 3. Convert both to pandas and merge
# =====================================================
df1 = all_ds1.to_pandas()
df2 = all_ds2.to_pandas()

df1["text"] = df1["text"].astype(str)
df1["label"] = df1["label"].astype(int)

df2["text"] = df2["text"].astype(str)
df2["label"] = df2["label"].astype(int)

merged_df = pd.concat([df1, df2], ignore_index=True)

print("before dedup:", len(merged_df))


# =====================================================
# 4. Clean and remove duplicates
# =====================================================
merged_df = merged_df.dropna(subset=["text", "label"])
merged_df["text"] = merged_df["text"].str.strip()
merged_df = merged_df[merged_df["text"] != ""]

# deduplicate by text only
merged_df = merged_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("after dedup:", len(merged_df))


# =====================================================
# 5. Create ordered id and keep only required columns
# =====================================================
merged_df.insert(0, "id", range(len(merged_df)))
merged_df = merged_df[["id", "text", "label"]]

# final validation: must match project loader requirements
allowed_labels = {0, 1, 2}
found_labels = set(merged_df["label"].unique().tolist())
unexpected = found_labels - allowed_labels
if unexpected:
    raise ValueError(f"Unexpected labels found: {sorted(unexpected)}")

print(merged_df.head())
print("final label counts:")
print(merged_df["label"].value_counts().sort_index())

print("label meaning:")
print("0 = negative")
print("1 = positive")
print("2 = neutral")


# =====================================================
# 6. Save
# =====================================================
merged_df.to_csv("merged_sentiment_clean.csv", index=False, encoding="utf-8-sig")
print("Saved: merged_sentiment_clean.csv")