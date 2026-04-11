"""
Baseline B: TF-IDF + Logistic Regression
Improvements over plain unigram TF-IDF:
  - Bigrams (ngram_range=(1,2)) to capture phrases like "not good"
  - Sublinear TF scaling to reduce high-frequency word dominance
  - min_df=3 to filter rare noise terms
  - class_weight='balanced' to handle class imbalance
  - solver='saga' supports n_jobs=-1 for parallel training

Label mapping:
    0 = negative | 1 = positive | 2 = neutral
"""

import re
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Preprocessing
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
# Model builder
# ---------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=80000,
        sublinear_tf=True,
        min_df=3,
        strip_accents='unicode'
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        solver='saga',
        multi_class='multinomial',
        random_state=42,
        n_jobs=-1
    )


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
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix - Baseline B (TF-IDF + LR)')
        plt.tight_layout()
        plt.savefig('baselineB_confusion_matrix.png', dpi=150)
        plt.close()
        print("Saved: baselineB_confusion_matrix.png")

    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Macro F1':    f1_score(y_true, y_pred, average='macro'),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted'),
    }


def print_top_features(vectorizer: TfidfVectorizer, clf: LogisticRegression, n: int = 15):
    feature_names = vectorizer.get_feature_names_out()
    for i, cls in enumerate(['negative', 'positive', 'neutral']):
        top_idx = np.argsort(clf.coef_[i])[-n:][::-1]
        print(f"Top features [{cls}]:")
        print("  ", ', '.join(feature_names[top_idx]))
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Load data ---
    df = pd.read_csv('merged_sentiment_clean.csv', encoding='utf-8-sig')
    print(f"Total rows: {len(df)}")
    print(df['label'].value_counts().sort_index()
          .rename({0: 'negative', 1: 'positive', 2: 'neutral'}))

    # --- Preprocess ---
    df['clean_text'] = df['text'].apply(preprocess)
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")

    # --- Split ---
    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df['label']
    )
    X_clean_train = df.loc[idx_train, 'clean_text']
    X_clean_test  = df.loc[idx_test,  'clean_text']
    y_train       = df.loc[idx_train, 'label']
    y_test        = df.loc[idx_test,  'label']
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")

    # --- Vectorise ---
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_clean_train)
    X_test_vec  = vectorizer.transform(X_clean_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train matrix shape: {X_train_vec.shape}")

    # --- Train ---
    t0 = time.time()
    clf = build_classifier()
    clf.fit(X_train_vec, y_train)
    print(f"Training time: {time.time() - t0:.1f}s")

    # --- Evaluate ---
    y_pred = clf.predict(X_test_vec)
    metrics = evaluate(y_test, y_pred, save_cm=True)

    # --- Top features ---
    print_top_features(vectorizer, clf)

    # --- Quick inference demo ---
    label_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    samples = [
        "I love this product, it's amazing!",
        "This is absolutely terrible, worst experience ever.",
        "The package arrived on Tuesday.",
        "Not bad, could be better.",
    ]
    preds = clf.predict(vectorizer.transform([preprocess(s) for s in samples]))
    print(f"{'Text':<50} {'Prediction':>12}")
    print('-' * 63)
    for text, p in zip(samples, preds):
        print(f"{text:<50} {label_map[p]:>12}")

    return metrics


if __name__ == '__main__':
    main()
