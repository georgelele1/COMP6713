from datasets import load_dataset, concatenate_datasets
import pandas as pd

# =====================================================
# 1. Load many_emotions
# target label:
# 0 = negative
# 1 = positive
# 2 = neutral
# =====================================================
ds1 = load_dataset("ma2za/many_emotions", "split")

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
    return {
        "text": example["text"],
        "label": int(many_emotions_map[int(example["label"])])
    }

all_ds1 = concatenate_datasets([
    ds1["train"],
    ds1["validation"],
    ds1["test"]
])

all_ds1 = all_ds1.map(
    remap_many_emotions,
    remove_columns=all_ds1.column_names
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
    return {
        "text": example["text"],
        "label": int(tweet_map[int(example["label"])])
    }

all_splits_ds2 = [ds2[split] for split in ds2.keys()]
all_ds2 = concatenate_datasets(all_splits_ds2)

all_ds2 = all_ds2.map(
    remap_tweet_sentiment,
    remove_columns=all_ds2.column_names
)

print("tweet_sentiment columns:", all_ds2.column_names)
print("tweet_sentiment rows:", len(all_ds2))


# =====================================================
# 3. Convert both to pandas first, then merge
# =====================================================
df1 = all_ds1.to_pandas()
df2 = all_ds2.to_pandas()

# force same types
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

# remove duplicate rows by text only
merged_df = merged_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print("after dedup:", len(merged_df))


# =====================================================
# 5. Create ordered id and keep only 3 columns
# =====================================================
merged_df.insert(0, "id", range(len(merged_df)))
merged_df = merged_df[["id", "text", "label"]]

print(merged_df.head())


# =====================================================
# 6. Save
# =====================================================
merged_df.to_csv("merged_sentiment_clean.csv", index=False, encoding="utf-8-sig")
print("Saved: merged_sentiment_clean.csv")