# """
# Predict fake vs real using TF-IDF + Logistic Regression model.
# """

import os
import joblib
from typing import Dict, List
from src.preprocess import concat_and_clean

DEFAULT_MODEL_PATH = os.environ.get("TFIDF_MODEL_PATH", "models/tfidf/model.pkl")

def load_model(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"TF-IDF model not found at {path}. Train and save first.")
    return joblib.load(path)

def predict_text(text: str, model=None) -> Dict:
    if model is None:
        model = load_model()
    cleaned = concat_and_clean('', text)
    pred = int(model.predict([cleaned])[0])
    prob = float(model.predict_proba([cleaned])[0][1])
    return {"prediction": pred, "probability": prob}

def predict_from_article(article: dict, model=None) -> Dict:
    if model is None:
        model = load_model()
    title = article.get("title","") or ""
    desc = article.get("description","") or ""
    content = article.get("content","") or ""
    text = " ".join([title, desc, content])
    cleaned = concat_and_clean(title, text)
    pred = int(model.predict([cleaned])[0])
    prob = float(model.predict_proba([cleaned])[0][1])
    return {
        "title": title,
        "url": article.get("url",""),
        "source": article.get("source",""),
        "prediction": pred,
        "probability": prob
    }

def predict_from_article_list(articles: List[dict], model=None) -> List[Dict]:
    if model is None:
        model = load_model()
    return [predict_from_article(a, model=model) for a in articles]
