"""
dataset_analysis.py
-------------------
Downloads multilingual sentiment datasets, normalises labels, detects
languages, and prints / saves detailed statistics.

Unified label scheme
--------------------
  0 = negative
  1 = positive
  2 = neutral

Dependencies
------------
    pip install datasets pandas langdetect matplotlib seaborn tqdm
"""

import warnings
warnings.filterwarnings("ignore")

import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

# Language detection – graceful fallback when langdetect is not installed
try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 42          # make results reproducible
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARN] langdetect not installed – language detection skipped.")
    print("       Install with:  pip install langdetect\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "negative", 1: "positive", 2: "neutral"}

def detect_language(text: str) -> str:
    """Return ISO-639-1 language code, or 'unknown'."""
    if not LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        return detect(str(text))
    except (LangDetectException, Exception):
        return "unknown"


def basic_clean(text: str) -> str:
    """Remove URLs and extra whitespace only – preserve non-ASCII characters."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders  (add more blocks here to extend)
# ─────────────────────────────────────────────────────────────────────────────

def load_many_emotions() -> pd.DataFrame:
    """
    ma2za/many_emotions  –  multilingual emotion dataset
    Original emotion labels  →  unified sentiment labels
        anger(0), fear(1)   → negative (0)
        joy(2), love(3)     → positive (1)
        sadness(4)          → negative (0)
        surprise(5)         → neutral  (2)   # ambiguous; treated as neutral
        neutral(6)          → neutral  (2)
    """
    print("  Downloading ma2za/many_emotions …")
    ds = load_dataset("ma2za/many_emotions", "split")

    emotion_to_sentiment = {
        0: 0,  # anger    → negative
        1: 0,  # fear     → negative
        2: 1,  # joy      → positive
        3: 1,  # love     → positive
        4: 0,  # sadness  → negative
        5: 2,  # surprise → neutral
        6: 2,  # neutral  → neutral
    }

    # original emotion label names (for the per-dataset report)
    emotion_names = {0: "anger", 1: "fear", 2: "joy", 3: "love",
                     4: "sadness", 5: "surprise", 6: "neutral"}

    splits = concatenate_datasets([ds[s] for s in ds.keys()])
    df = splits.to_pandas()[["text", "label"]].copy()
    df["original_label_name"] = df["label"].map(emotion_names)
    df["label"] = df["label"].map(emotion_to_sentiment).astype(int)
    df["source"] = "many_emotions"
    return df


def load_tweet_sentiment() -> pd.DataFrame:
    """
    mteb/tweet_sentiment_extraction  –  English tweets
    Original labels  →  unified
        negative(0) → 0
        neutral(1)  → 2   (swap: original 1=neutral, we want 2=neutral)
        positive(2) → 1   (swap: original 2=positive, we want 1=positive)
    """
    print("  Downloading mteb/tweet_sentiment_extraction …")
    ds = load_dataset("mteb/tweet_sentiment_extraction")

    tweet_map = {0: 0, 1: 2, 2: 1}

    splits = concatenate_datasets([ds[s] for s in ds.keys()])
    df = splits.to_pandas()[["text", "label"]].copy()
    df["original_label_name"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})
    df["label"] = df["label"].map(tweet_map).astype(int)
    df["source"] = "tweet_sentiment"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def per_dataset_stats(df: pd.DataFrame, name: str) -> None:
    """Print label distribution and text-length stats for a single dataset."""
    print(f"\n{'─'*60}")
    print(f"  Dataset : {name}")
    print(f"  Rows    : {len(df):,}")

    # label distribution
    lc = df["label"].value_counts().sort_index()
    print(f"\n  Label distribution:")
    for lbl, cnt in lc.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {LABEL_NAMES[lbl]:10s}  {cnt:7,}  ({pct:5.1f}%)  {bar}")

    # original label breakdown (if available)
    if "original_label_name" in df.columns:
        print(f"\n  Original label → unified label mapping:")
        for orig, grp in df.groupby("original_label_name"):
            unified = LABEL_NAMES[grp["label"].iloc[0]]
            print(f"    {orig:12s} → {unified}  ({len(grp):,})")

    # text length
    lengths = df["text"].str.len()
    print(f"\n  Text length (chars):")
    print(f"    min={lengths.min()}, median={lengths.median():.0f}, "
          f"mean={lengths.mean():.0f}, max={lengths.max()}")
    print(f"{'─'*60}")


def language_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Print and visualise language distribution."""
    if not LANGDETECT_AVAILABLE:
        return

    print("\n  Detecting languages (this may take a moment) …")
    tqdm.pandas(desc="  lang-detect")
    df["lang"] = df["text"].progress_apply(detect_language)

    total = len(df)
    lang_counts = df["lang"].value_counts()

    print(f"\n  Language distribution (top 15):")
    for lang, cnt in lang_counts.head(15).items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {lang:8s}  {cnt:7,}  ({pct:5.1f}%)  {bar}")

    # label × language cross-table
    print(f"\n  Label distribution per language (top 8 languages):")
    top_langs = lang_counts.head(8).index.tolist()
    sub = df[df["lang"].isin(top_langs)]
    ct = pd.crosstab(sub["lang"], sub["label"].map(LABEL_NAMES), margins=True)
    print(ct.to_string())

    return df  # with "lang" column added (always returns df)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame, title: str, filename: str) -> None:
    counts = df["label"].map(LABEL_NAMES).value_counts().reindex(
        ["negative", "positive", "neutral"], fill_value=0)
    colors = ["#e74c3c", "#2ecc71", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(val):,}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_language_distribution(df: pd.DataFrame, filename: str, top_n: int = 12) -> None:
    if "lang" not in df.columns:
        return
    lang_counts = df["lang"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=lang_counts.index, y=lang_counts.values, ax=ax,
                palette="viridis", edgecolor="white")
    for bar, val in zip(ax.patches, lang_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Language Distribution (top {top_n})", fontsize=13, pad=10)
    ax.set_xlabel("Language code")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_label_per_language(df: pd.DataFrame, filename: str, top_n: int = 8) -> None:
    if "lang" not in df.columns:
        return
    top_langs = df["lang"].value_counts().head(top_n).index.tolist()
    sub = df[df["lang"].isin(top_langs)].copy()
    sub["sentiment"] = sub["label"].map(LABEL_NAMES)

    ct = (sub.groupby(["lang", "sentiment"])
             .size()
             .reset_index(name="count"))

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = ct.pivot(index="lang", columns="sentiment", values="count").fillna(0)
    # reorder columns
    pivot = pivot.reindex(columns=["negative", "positive", "neutral"])
    pivot.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71", "#95a5a6"],
               edgecolor="white", width=0.7)
    ax.set_title(f"Sentiment per Language (top {top_n} languages)", fontsize=13, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Sentiment")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_text_length_distribution(df: pd.DataFrame, filename: str) -> None:
    df = df.copy()
    df["length"] = df["text"].str.len().clip(upper=500)   # cap outliers for readability
    df["sentiment"] = df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"negative": "#e74c3c", "positive": "#2ecc71", "neutral": "#95a5a6"}
    for lbl, grp in df.groupby("sentiment"):
        grp["length"].plot.hist(ax=ax, bins=50, alpha=0.55,
                                label=lbl, color=colors[lbl])
    ax.set_title("Text Length Distribution by Sentiment", fontsize=13, pad=10)
    ax.set_xlabel("Character count (capped at 500)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 1 – Downloading datasets")
    print("=" * 60)

    df_many   = load_many_emotions()
    df_tweet  = load_tweet_sentiment()

    raw_frames = {"many_emotions": df_many, "tweet_sentiment": df_tweet}

    # ── Per-dataset stats (before merge) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 – Per-dataset statistics (before merge)")
    print("=" * 60)

    for name, df in raw_frames.items():
        per_dataset_stats(df, name)

    # ── Merge ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 – Merging & cleaning")
    print("=" * 60)

    merged = pd.concat(raw_frames.values(), ignore_index=True)
    print(f"  Rows after concat      : {len(merged):,}")

    # basic cleaning
    merged["text"] = merged["text"].apply(basic_clean)
    merged = merged[merged["text"].str.len() > 0]
    print(f"  Rows after empty drop  : {len(merged):,}")

    # deduplicate by text (keep first occurrence)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  Duplicates removed     : {before_dedup - len(merged):,}")
    print(f"  Final rows             : {len(merged):,}")

    # ── Merged dataset stats ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 – Merged dataset statistics")
    print("=" * 60)

    per_dataset_stats(merged, "MERGED")

    # source breakdown
    print(f"\n  Source breakdown:")
    sc = merged["source"].value_counts()
    for src, cnt in sc.items():
        print(f"    {src:25s}  {cnt:7,}  ({cnt/len(merged)*100:.1f}%)")

    # ── Language detection ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 – Language detection")
    print("=" * 60)

    result = language_stats(merged)
    if result is not None:
        merged = result

    # ── Visualisations ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 – Saving plots")
    print("=" * 60)

    plot_label_distribution(merged, "Merged Dataset – Label Distribution",
                            "stats_label_distribution.png")
    plot_text_length_distribution(merged, "stats_text_length.png")

    if "lang" in merged.columns:
        plot_language_distribution(merged, "stats_language_distribution.png")
        plot_label_per_language(merged,    "stats_label_per_language.png")

    # ── Processing pipeline ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7 – Processing pipeline")
    print("=" * 60)

    processed = merged.copy()

    # 7a. (removed) English-only filter was dropped — we use bert-base-multilingual-uncased
    #     which supports 104 languages. Keeping all languages maximises training data
    #     and lets the model leverage its multilingual pre-training.

    # 7b. Clean text: remove URLs, @mentions, #hashtag symbol
    #     Keep punctuation (!, ?, ...) — useful for sentiment
    def clean_text(text: str) -> str:
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)   # URLs
        text = re.sub(r"@\w+", "", text)              # @mentions
        text = re.sub(r"#(\w+)", r"\1", text)         # #tag → tag (keep word)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    processed["text"] = processed["text"].apply(clean_text)

    # 7c. Minimum length filter: at least 3 words
    before = len(processed)
    processed = processed[
        processed["text"].str.split().str.len() >= 3
    ].reset_index(drop=True)
    print(f"  [7c] Min-length filter  : {before:,} → {len(processed):,}"
          f"  (removed {before - len(processed):,} short rows)")

    # 7d. Drop any new empty / duplicate rows after cleaning
    before = len(processed)
    processed = processed.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"  [7d] Re-dedup           : {before:,} → {len(processed):,}"
          f"  (removed {before - len(processed):,} duplicates)")

    # 7e. Undersample to balance labels (cap each class at min-class count)
    print(f"\n  Label counts before balancing:")
    for lbl in sorted(processed["label"].unique()):
        cnt = (processed["label"] == lbl).sum()
        print(f"    {LABEL_NAMES[lbl]:10s}: {cnt:,}")

    min_count = processed["label"].value_counts().min()
    balanced_parts = [
        processed[processed["label"] == lbl].sample(min_count, random_state=42)
        for lbl in sorted(processed["label"].unique())
    ]
    processed = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Label counts after balancing (each capped at {min_count:,}):")
    for lbl in sorted(processed["label"].unique()):
        cnt = (processed["label"] == lbl).sum()
        print(f"    {LABEL_NAMES[lbl]:10s}: {cnt:,}")

    # 7f. Extract English-only subset before dropping the lang column
    #     (lang column is removed in the next step)
    if "lang" in processed.columns:
        en_processed = processed[processed["lang"] == "en"].reset_index(drop=True)
    else:
        en_processed = processed[
            ~processed["text"].str.contains(r"[^\x00-\x7F]", regex=True, na=False)
        ].reset_index(drop=True)

    # 7g. Final column selection and ID assignment
    keep_cols = ["text", "label", "source"]
    processed   = processed[keep_cols].copy()
    en_processed = en_processed[keep_cols].copy()
    processed.insert(0, "id", range(len(processed)))
    en_processed.insert(0, "id", range(len(en_processed)))

    # ── Save processed CSVs ──────────────────────────────────────────────────
    out_path = "sentiment_processed.csv"
    processed.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {out_path}  ({len(processed):,} rows)")

    en_path = "sentiment_processed_en.csv"
    en_processed.to_csv(en_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {en_path}  ({len(en_processed):,} rows, English only)")

    # ── Train / Val / Test split ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8 – Train / Val / Test split  (70 / 15 / 15)")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    # Step 1: split off 30 % (val + test) with stratification
    train_df, temp_df = train_test_split(
        processed,
        test_size=0.30,
        random_state=42,
        stratify=processed["label"],
    )

    # Step 2: split the 30 % equally into val and test (15 % each of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    print("\n  Multilingual splits (for mBERT):")
    for split_name, split_df in splits.items():
        out = f"sentiment_{split_name}.csv"
        split_df.to_csv(out, index=False, encoding="utf-8-sig")
        lc = split_df["label"].value_counts().sort_index()
        dist = "  |  ".join(
            f"{LABEL_NAMES[l]}: {c:,}" for l, c in lc.items()
        )
        print(f"  {split_name:5s}  {len(split_df):7,} rows   [{dist}]   → {out}")

    # English-only splits for baselines (TF-IDF, word counting)
    print("\n  English-only splits (for baselines):")
    from sklearn.model_selection import train_test_split as _tts
    en_train, en_temp = _tts(en_processed, test_size=0.30, random_state=42,
                              stratify=en_processed["label"])
    en_val, en_test   = _tts(en_temp,      test_size=0.50, random_state=42,
                              stratify=en_temp["label"])
    for split_name, split_df in [("train", en_train), ("val", en_val), ("test", en_test)]:
        split_df = split_df.reset_index(drop=True)
        out = f"sentiment_en_{split_name}.csv"
        split_df.to_csv(out, index=False, encoding="utf-8-sig")
        lc = split_df["label"].value_counts().sort_index()
        dist = "  |  ".join(
            f"{LABEL_NAMES[l]}: {c:,}" for l, c in lc.items()
        )
        print(f"  {split_name:5s}  {len(split_df):7,} rows   [{dist}]   → {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    lc = processed["label"].value_counts().sort_index()
    for lbl, cnt in lc.items():
        print(f"  {LABEL_NAMES[lbl]:10s}: {cnt:7,}  ({cnt/len(processed)*100:.1f}%)")
    print(f"\n  Total samples      : {len(processed):,}")
    print(f"  Output file        : sentiment_processed.csv  (multilingual, for mBERT)")
    print(f"  Split files        : sentiment_train/val/test.csv  (multilingual)")
    print(f"  English output     : sentiment_processed_en.csv")
    print(f"  English splits     : sentiment_en_train/val/test.csv  (for baselines)")
    print("=" * 60)


if __name__ == "__main__":
    main()
