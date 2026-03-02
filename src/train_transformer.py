"""
Train DistilBERT (transformers) on ISOT dataset.

Usage:
  python -m src.train_transformer --data_dir data/ISOT --extra_data "data/Mahdi Mashayekhi/fake_news_dataset.csv" --output_dir models/transformer --results_dir results/transformer --epochs 3 --batch_size 16

Notes:
- Recommended to run with GPU.
- Training time depends on dataset size and hardware.
"""

import os
import argparse
import random
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from datasets import Dataset
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# ensure src import works when running script from project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from src.data_utils import load_combined_dataset

SEED = 42
MODEL_NAME = "distilbert-base-uncased"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        if torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:
                pass
    except Exception:
        pass


def prepare_dataframe_for_training(df: pd.DataFrame, balance: bool = True):
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["text"] = df.get("text", "").fillna("").astype(str)
    df["text_all"] = (df["title"] + " " + df["text"]).str.strip()

    if balance:
        counts = df["label"].value_counts()
        min_count = int(counts.min())
        dfs = []
        for lbl in counts.index:
            dfs.append(df[df["label"] == lbl].sample(min_count, random_state=SEED))
        df_bal = pd.concat(dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
        return df_bal
    else:
        return df


def tokenize_fn(examples, tokenizer, max_length=256):
    return tokenizer(examples["text_all"], truncation=True, max_length=max_length)


def compute_metrics_fn(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


def compute_metrics_from_preds(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}


def save_plot_roc(y_true, y_probs, out_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return roc_auc

def save_confusion_matrix(cm, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Real", "Fake"])
    plt.yticks(tick_marks, ["Real", "Fake"])

    # Add cell numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def find_best_threshold_from_logits(logits, labels, metric="f1"):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    best_thresh, best_score = 0.5, 0.0
    thresholds = np.linspace(0.1, 0.9, 17)
    for t in thresholds:
        preds = (probs > t).astype(int)
        if metric == "f1":
            score = f1_score(labels, preds)
        else:
            score = accuracy_score(labels, preds)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score


def main(args):
    set_seed()
    print("Loading combined dataset...")
    df = load_combined_dataset(args.data_dir, args.extra_data)

    df = prepare_dataframe_for_training(df, balance=True)
    print("Post-balance counts:", df["label"].value_counts().to_dict())

    # Stratified split
    from sklearn.model_selection import train_test_split as _t
    train_df, val_df = _t(df, test_size=args.val_split, stratify=df["label"], random_state=SEED)
    print("Train / Val sizes:", len(train_df), len(val_df))

    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    # Tokenize
    def map_fn(batch):
        return tokenize_fn(batch, tokenizer, max_length=args.max_length)

    tokenized_train = ds_train.map(map_fn, batched=True, remove_columns=["title", "text"])
    tokenized_val = ds_val.map(map_fn, batched=True, remove_columns=["title", "text"])

    tokenized_train = tokenized_train.add_column("labels", ds_train["label"])
    tokenized_val = tokenized_val.add_column("labels", ds_val["label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # device info only (Trainer handles placement)
    device_name = "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    print("Using device:", device_name)

    # === Simple (stable) TrainingArguments for best comparable performance (Choice A) ===
    # Use argument names compatible with your installed transformers (observed: eval_strategy etc.)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        eval_strategy="epoch",   # compatible name in your Transformers version
        save_strategy="epoch",   # compatible name
        logging_strategy="steps", # ensures logging_steps is used
    )

    # Trainer with compute_metrics to mirror first script's training behavior
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training done.")

    # === Evaluate on validation set (use trainer.predict to get logits) ===
    print("Running predictions on validation set to compute metrics...")
    preds_output = trainer.predict(tokenized_val)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds_argmax = np.argmax(logits, axis=-1)

    # Find best threshold based on logits (F1)
    best_thresh, best_score = find_best_threshold_from_logits(logits, labels, metric="f1")
    print(f"Best threshold: {best_thresh:.3f} (score {best_score:.4f})")

    # Prepare results directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save threshold in both model folder and results folder
    thresh_path_model = os.path.join(args.output_dir, "best_threshold.txt")
    with open(thresh_path_model, "w") as f:
        f.write(str(best_thresh))

    thresh_path_results = os.path.join(args.results_dir, "threshold.txt")
    with open(thresh_path_results, "w") as f:
        f.write(str(best_thresh))

    # Compute final metrics at chosen threshold
    final_metrics = compute_metrics_from_preds(labels, probs, threshold=best_thresh)
    # Save numeric metrics to text file
    metrics_text = (
        f"Accuracy: {final_metrics['accuracy']:.4f}\n"
        f"Precision: {final_metrics['precision']:.4f}\n"
        f"Recall: {final_metrics['recall']:.4f}\n"
        f"F1: {final_metrics['f1']:.4f}\n"
        f"ROC-AUC: {final_metrics['roc_auc']:.4f}\n"
    )
    with open(os.path.join(args.results_dir, "performance_metrics.txt"), "w") as f:
        f.write(metrics_text)

    # Classification report (use preds at threshold)
    preds_thresholded = (probs >= best_thresh).astype(int)
    report = classification_report(labels, preds_thresholded, target_names=["Real", "Fake"], digits=4)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(labels, preds_thresholded)
    save_confusion_matrix(cm, os.path.join(args.results_dir, "confusion_matrix.png"))

    # ROC curve
    try:
        roc_auc = save_plot_roc(labels, probs, os.path.join(args.results_dir, "roc_auc.png"), title="DistilBERT ROC")
    except Exception:
        roc_auc = None

    # Save model & tokenizer (Trainer.save_model is simple & recommended)
    print("Saving model and tokenizer to:", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training loss curve if available from trainer.state.log_history
    try:
        history = trainer.state.log_history if hasattr(trainer.state, "log_history") else None
        if history:
            steps = []
            losses = []
            for e in history:
                if "loss" in e:
                    steps.append(e.get("step", len(steps)))
                    losses.append(e["loss"])
            if len(losses) > 0:
                plt.figure(figsize=(7, 4))
                plt.plot(steps, losses, marker="o", lw=1)
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title("Training loss")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(args.results_dir, "loss_curve.png"), dpi=200)
                plt.close()
    except Exception:
        pass

    # Save metadata
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "model_dir": args.output_dir,
        "results_dir": args.results_dir,
        "best_threshold": float(best_thresh),
        "best_threshold_score": float(best_score),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }
    with open(os.path.join(args.results_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Transformer training complete. Results saved to:", args.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ISOT")
    parser.add_argument("--extra_data", type=str, default="data/Mahdi Mashayekhi/fake_news_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="models/transformer")
    parser.add_argument("--results_dir", type=str, default="results/transformer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()
    main(args)