"""
Train and save a TF-IDF + LogisticRegression pipeline on combined datasets (ISOT + ).

Usage:
  python -m src.train --data_dir data --extra_data data/fake_news_dataset.csv --out_model models/tfidf/model.pkl
"""
import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

# ensure your src is importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.preprocess import concat_and_clean
from src.data_utils import load_combined_dataset

RANDOM_SEED = 42

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['text_all'] = df.apply(lambda r: concat_and_clean(r['title'], r['text']), axis=1)
    return df

def save_plot_roc(y_true, y_probs, out_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return roc_auc

def save_confusion_matrix(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Real", "Fake"])
    plt.yticks(tick_marks, ["Real", "Fake"])

    # display cell values
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def compute_and_save_metrics(y_true, y_pred, y_probs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    accuracy = accuracy_score(y_true, y_pred)
    precision_real = precision_score(y_true, y_pred, pos_label=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0)
    f1_real = f1_score(y_true, y_pred, pos_label=0)

    precision_fake = precision_score(y_true, y_pred, pos_label=1)
    recall_fake = recall_score(y_true, y_pred, pos_label=1)
    f1_fake = f1_score(y_true, y_pred, pos_label=1)

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Real News - Precision: {precision_real:.4f}, Recall: {recall_real:.4f}, F1: {f1_real:.4f}\n"
        f"Fake News - Precision: {precision_fake:.4f}, Recall: {recall_fake:.4f}, F1: {f1_fake:.4f}\n"
    )

    with open(os.path.join(out_dir, "performance_metrics.txt"), "w") as f:
        f.write(metrics_text)

    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"))

    # ROC curve and AUC
    try:
        roc_auc = save_plot_roc(y_true, y_probs, os.path.join(out_dir, "roc_auc.png"), title="TF-IDF ROC Curve")
    except Exception:
        roc_auc = None

    return {
        "accuracy": float(accuracy),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "roc_auc": float(roc_auc) if roc_auc is not None else None
    }

def extract_feature_importances(pipeline, out_path, top_k=200):
    """
    For logistic regression, coef_ indicates weight per feature.
    We map top positive weights (fake-leaning) and negative (real-leaning).
    Save a JSON with {"fake": [[word,score],...], "real": [[word,score],...]}
    """
    vect = pipeline.named_steps.get("tfidf", None)
    clf = pipeline.named_steps.get("clf", None)
    if vect is None or clf is None:
        return None
    vocab = vect.vocabulary_
    # inverse mapping: index -> token
    inv = {idx: token for token, idx in vocab.items()}
    coefs = clf.coef_.flatten()
    pairs = []
    for idx, score in enumerate(coefs):
        token = inv.get(idx, None)
        if token:
            pairs.append((token, float(score)))
    # sort by score
    sorted_by_score = sorted(pairs, key=lambda x: -x[1])
    fake_top = sorted_by_score[:top_k]
    real_top = sorted(sorted_by_score, key=lambda x: x[1])[:top_k]
    out = {"fake": fake_top, "real": real_top}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out

def train_and_save(data_dir, extra_data, model_out_path, results_dir, random_seed=RANDOM_SEED):
    # load combined dataset
    print("Loading combined dataset...")
    df = load_combined_dataset(data_dir, extra_data)
    print(f"Dataset size: {len(df)}")
    df = prepare_data(df)

    X = df['text_all']
    y = df['label']

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    ])

    print("Training TF-IDF + LogisticRegression pipeline...")
    pipeline.fit(X_train, y_train)

    # predict
    y_pred = pipeline.predict(X_valid)
    y_prob = pipeline.predict_proba(X_valid)[:, 1]

    # metrics and plots
    os.makedirs(results_dir, exist_ok=True)
    metrics = compute_and_save_metrics(y_valid, y_pred, y_prob, results_dir)

    # save feature importances
    fi = extract_feature_importances(pipeline, os.path.join(results_dir, "feature_importances.json"), top_k=300)

    # save model to cleaner path models/tfidf/model.pkl
    model_dir = os.path.dirname(model_out_path)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_out_path)
    print(f"Saved TF-IDF model to {model_out_path}")

    # save some metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat()+"Z",
        "model_path": model_out_path,
        "results_dir": results_dir,
        "metrics": metrics
    }
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("TF-IDF training complete. Results saved to:", results_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ISOT")
    parser.add_argument("--extra_data", type=str, default="data/Mahdi Mashayekhi/fake_news_dataset.csv")
    parser.add_argument("--out_model", type=str, default="models/tfidf/model.pkl")
    parser.add_argument("--results_dir", type=str, default="results/tfidf")
    args = parser.parse_args()

    train_and_save(args.data_dir, args.extra_data, args.out_model, args.results_dir)
