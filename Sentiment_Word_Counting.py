"""
Baseline A: Sentiment Word Counting (rule-based, English only)
--------------------------------------------------------------
Uses the NLTK Opinion Lexicon (Bing Liu). Rule: count positive and
negative words; majority wins, tie → neutral.

Label mapping:
    0 = negative | 1 = positive | 2 = neutral
"""

import pandas as pd
import re

import nltk
nltk.download('opinion_lexicon', quiet=True)
from nltk.corpus import opinion_lexicon

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
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, label: str, save_cm: bool = False):
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
        ax.set_title(f'Confusion Matrix - Baseline A (Word Counting) [{label}]')
        plt.tight_layout()
        fname = f'baselineA_confusion_matrix.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Macro F1':    f1_score(y_true, y_pred, average='macro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load pre-split English-only data ---
    val_df  = pd.read_csv('sentiment_en_val.csv',  encoding='utf-8-sig')
    test_df = pd.read_csv('sentiment_en_test.csv', encoding='utf-8-sig')

    print(f"Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print("Label distribution (test):")
    print(test_df['label'].value_counts().sort_index()
          .rename({0: 'negative', 1: 'positive', 2: 'neutral'}))

    # --- Predict (rule-based, no training needed) ---
    y_val_pred  = val_df['text'].apply(predict_word_count)
    y_test_pred = test_df['text'].apply(predict_word_count)

    print(f"\nPrediction distribution (test): "
          f"{y_test_pred.value_counts().sort_index().to_dict()}")

    # --- Evaluate ---
    print("\n=== Validation set ===")
    evaluate(val_df['label'], y_val_pred, label='val', save_cm=False)

    print("\n=== Test set ===")
    metrics = evaluate(test_df['label'], y_test_pred, label='test', save_cm=True)

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
