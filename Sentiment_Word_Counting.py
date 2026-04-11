"""
Baseline A: Sentiment Word Counting (rule-based)
Uses the NLTK Opinion Lexicon (Bing Liu).

Label mapping:
    0 = negative | 1 = positive | 2 = neutral
"""

import pandas as pd
import re

import nltk
nltk.download('opinion_lexicon', quiet=True)
from nltk.corpus import opinion_lexicon

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
POS_WORDS = set(opinion_lexicon.positive())
NEG_WORDS = set(opinion_lexicon.negative())


# ---------------------------------------------------------------------------
# Preprocessing (kept for interface consistency; Baseline A uses raw text)
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_word_count(text: str) -> int:
    """Return 0 (negative), 1 (positive), or 2 (neutral)."""
    tokens = str(text).lower().split()
    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    if pos > neg:
        return 1
    elif neg > pos:
        return 0
    else:
        return 2


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, save_cm: bool = True):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=['negative', 'positive', 'neutral']
    ))

    if save_cm:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['negative', 'positive', 'neutral']
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap='Oranges')
        ax.set_title('Confusion Matrix - Baseline A (Word Counting)')
        plt.tight_layout()
        plt.savefig('baselineA_confusion_matrix.png', dpi=150)
        plt.close()
        print("Saved: baselineA_confusion_matrix.png")

    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Macro F1':    f1_score(y_true, y_pred, average='macro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load data ---
    df = pd.read_csv('merged_sentiment_clean.csv', encoding='utf-8-sig')
    print(f"Total rows: {len(df)}")
    print(df['label'].value_counts().sort_index()
          .rename({0: 'negative', 1: 'positive', 2: 'neutral'}))

    # --- Preprocess (used only to drop empty rows; Baseline A uses raw text) ---
    df['clean_text'] = df['text'].apply(preprocess)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")

    # --- Split ---
    _, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df['label']
    )
    X_raw_test = df.loc[idx_test, 'text']
    y_test     = df.loc[idx_test, 'label']

    # --- Predict ---
    y_pred = X_raw_test.apply(predict_word_count)
    print("Prediction distribution:", y_pred.value_counts().sort_index().to_dict())

    # --- Evaluate ---
    metrics = evaluate(y_test, y_pred, save_cm=True)

    # --- Quick inference demo ---
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    samples = [
        "I love this product, it's amazing!",
        "This is absolutely terrible, worst experience ever.",
        "The package arrived on Tuesday.",
        "Not bad, could be better.",
    ]
    preds = [predict_word_count(s) for s in samples]
    print(f"\n{'Text':<50} {'Prediction':>12}")
    print('-' * 63)
    for text, p in zip(samples, preds):
        print(f"{text:<50} {label_map[p]:>12}")

    return metrics


if __name__ == '__main__':
    main()
