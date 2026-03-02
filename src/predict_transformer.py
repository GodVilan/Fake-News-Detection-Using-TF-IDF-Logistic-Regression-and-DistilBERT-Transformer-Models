# """
# Prediction utilities using the fine-tuned DistilBERT model.

# Usage examples:
#   from src.predict_transformer import load_transformer, predict_text, predict_from_article_list
#   model, tokenizer, device = load_transformer("models/transformer")
#   predict_text("Some headline", model, tokenizer, device)
# """

import os
from typing import Dict, List, Tuple
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import concat_and_clean

DEFAULT_MODEL_DIR = os.environ.get("TRANSFORMER_MODEL_DIR", "models/transformer")
DEFAULT_THRESHOLD = 0.6

def load_best_threshold(model_dir: str = DEFAULT_MODEL_DIR) -> float:
    # prefer model_dir/best_threshold.txt then results/transformer/threshold.txt
    p1 = os.path.join(model_dir, "best_threshold.txt")
    p2 = os.path.join("results", "transformer", "threshold.txt")
    for p in (p1, p2):
        if os.path.exists(p):
            try:
                return float(open(p, "r").read().strip())
            except Exception:
                continue
    return DEFAULT_THRESHOLD

def load_transformer(model_dir: str = DEFAULT_MODEL_DIR):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Transformer model dir not found at {model_dir}. Train and save first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    return model, tokenizer, device

def predict_text(text: str, model=None, tokenizer=None, device=None, max_length: int = 256) -> Dict:
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_transformer()
    cleaned = concat_and_clean("", text)
    if len(cleaned.split()) < 5:
        return {"prediction": -1, "probability_fake": 0.0, "prob_real": 0.0}
    inputs = tokenizer(cleaned, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()
        probs = F.softmax(logits, dim=-1).numpy()[0]
        pred = int(np.argmax(probs))
        prob_fake = float(probs[1])
        prob_real = float(probs[0])
    return {"prediction": pred, "probability_fake": prob_fake, "prob_real": prob_real}

def predict_from_article(article: dict, model=None, tokenizer=None, device=None) -> Dict:
    title = article.get("title","") or ""
    desc = article.get("description","") or ""
    content = article.get("content","") or ""
    text = " ".join([title, desc, content]).strip()
    out = predict_text(text, model=model, tokenizer=tokenizer, device=device)
    # apply threshold saved
    threshold = load_best_threshold()
    label = -1
    if out["prediction"] != -1:
        label = 1 if out["probability_fake"] > threshold else 0
    return {"title": title, "url": article.get("url",""), "source": article.get("source",""), "prediction": int(label), "probability_fake": float(out.get("probability_fake", 0.0))}

def predict_from_article_list(articles: List[dict], model=None, tokenizer=None, device=None) -> List[Dict]:
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_transformer()
    results = []
    for a in articles:
        results.append(predict_from_article(a, model=model, tokenizer=tokenizer, device=device))
    return results

def get_current_threshold(model_dir: str = DEFAULT_MODEL_DIR) -> float:
    return load_best_threshold(model_dir)