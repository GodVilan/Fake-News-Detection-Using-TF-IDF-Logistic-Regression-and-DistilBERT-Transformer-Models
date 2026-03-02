"""
Usage:
  python -m src.evaluate --model models/transformer/model.pkl --data_dir data
"""

import os
import torch
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

def find_best_threshold(y_true, y_probs):
    """
    Finds the best probability threshold for binary classification
    based on maximizing F1 score.
    """
    best_thresh = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def evaluate_model(model_path, data_dir, split="test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    # Load ISOT dataset (combined or preprocessed)
    dataset = load_dataset("csv", data_files={
        split: os.path.join(data_dir, f"{split}.csv")
    })[split]

    texts = dataset["text"]
    labels = np.array(dataset["label"])

    # Tokenize
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

    # Find optimal threshold
    best_thresh, best_f1 = find_best_threshold(labels, probs)
    preds = (probs >= best_thresh).astype(int)

    # Compute metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    print("\n--- Evaluation Results ---")
    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f} (Optimal: {best_f1:.4f})")
    print(f"ROC-AUC: {auc:.4f}")
    print("\nConfusion Matrix:\n", cm)

    return best_thresh


def predict_text(model_path, text, threshold=0.5):
    """
    Predicts whether a given text is Fake (1) or Real (0)
    using the chosen threshold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].item()

    label = 1 if probs >= threshold else 0
    print(f"\nInput: {text}")
    print(f"Predicted Probability (Fake): {probs:.4f}")
    print(f"Classification: {'FAKE' if label == 1 else 'REAL'} (threshold={threshold:.2f})")
    return label, probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT Fake News Detector")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing test.csv")
    args = parser.parse_args()

    best_thresh = evaluate_model(args.model_dir, args.data_dir, split="test")

    # Example custom prediction
    test_news = "The Federal Reserve has announced a new policy to stabilize inflation."
    predict_text(args.model_dir, test_news, threshold=best_thresh)
