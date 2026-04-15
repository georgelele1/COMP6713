"""
Baseline B: Character n-gram TF-IDF + Logistic Regression (Multilingual)
-------------------------------------------------------------------------
Uses character-level n-grams (2–4) instead of word tokens, which makes
the model language-agnostic — no tokenisation or external lexicon needed.

Label mapping:
    0 = negative | 1 = positive | 2 = neutral
"""

import re
import time

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Light cleaning — preserve non-ASCII characters for multilingual support."""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)         # #tag → tag
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer='char_wb',       # character n-grams (language-agnostic)
        ngram_range=(2, 4),       # bi- to 4-grams
        max_features=80000,
        sublinear_tf=True,
        min_df=3,
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='saga',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Evaluation
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
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix - Baseline B (Char n-gram TF-IDF + LR)')
        plt.tight_layout()
        plt.savefig('baselineB_confusion_matrix.png', dpi=150)
        plt.close()
        print("Saved: baselineB_confusion_matrix.png")

    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Macro F1':    f1_score(y_true, y_pred, average='macro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load pre-split multilingual data ---
    train_df = pd.read_csv('sentiment_train.csv', encoding='utf-8-sig')
    val_df   = pd.read_csv('sentiment_val.csv',   encoding='utf-8-sig')
    test_df  = pd.read_csv('sentiment_test.csv',  encoding='utf-8-sig')

    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    print("Label distribution (train):")
    print(train_df['label'].value_counts().sort_index()
          .rename({0: 'negative', 1: 'positive', 2: 'neutral'}))

    # --- Preprocess ---
    for df in [train_df, val_df, test_df]:
        df['clean_text'] = df['text'].apply(preprocess)

    X_train = train_df['clean_text']
    X_val   = val_df['clean_text']
    X_test  = test_df['clean_text']
    y_train = train_df['label']
    y_val   = val_df['label']
    y_test  = test_df['label']

    # --- Vectorise ---
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec   = vectorizer.transform(X_val)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"\nVocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"Train matrix    : {X_train_vec.shape}")

    # --- Train ---
    t0 = time.time()
    clf = build_classifier()
    clf.fit(X_train_vec, y_train)
    print(f"Training time   : {time.time() - t0:.1f}s")

    # --- Validate ---
    print("\n=== Validation set ===")
    evaluate(y_val, clf.predict(X_val_vec), save_cm=False)

    # --- Test ---
    print("\n=== Test set ===")
    metrics = evaluate(y_test, clf.predict(X_test_vec), save_cm=True)

    # --- Quick inference demo ---
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    samples = [
        "I love this product, it's amazing!",
        "This is absolutely terrible, worst experience ever.",
        "The package arrived on Tuesday.",
        "Not bad, could be better.",
        "这个产品真的很棒！",          # Chinese
        "C'est vraiment décevant.",    # French
    ]
    preds = clf.predict(vectorizer.transform([preprocess(s) for s in samples]))
    print(f"\n{'Text':<45} {'Prediction':>12}")
    print('-' * 58)
    for text, p in zip(samples, preds):
        print(f"{text:<45} {label_map[p]:>12}")

    return metrics


if __name__ == '__main__':
    main()
