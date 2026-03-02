import os
import pandas as pd
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def load_isot_dataset(data_dir: str = "data/ISOT") -> pd.DataFrame:
    true_path = os.path.join(data_dir, "True.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")
    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError("Expecting ISOT dataset files True.csv and Fake.csv in data/ISOT/")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    df_true = df_true.rename(columns={c.lower(): c.lower() for c in df_true.columns})
    df_fake = df_fake.rename(columns={c.lower(): c.lower() for c in df_fake.columns})
    # Standardize columns
    def _ensure(df):
        cols = df.columns.tolist()
        title_col = "title" if "title" in cols else cols[0]
        text_col = "text" if "text" in cols else (cols[1] if len(cols) > 1 else cols[0])
        out = df[[title_col, text_col]].copy()
        out.columns = ["title", "text"]
        return out
    df_true = _ensure(df_true)
    df_fake = _ensure(df_fake)
    df_true["label"] = 0  # REAL
    df_fake["label"] = 1  # FAKE
    return pd.concat([df_true, df_fake], ignore_index=True)

def load_additional_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in Mahdi Mashayekhi dataset.")
    # Map textual labels -> numeric: real -> 0, fake -> 1
    df["label"] = df["label"].astype(str).str.lower().map({"real": 0, "fake": 1})
    # If any unmapped, drop or set default (here we'll drop nulls)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if "title" not in df.columns or "text" not in df.columns:
        # try picking nearest columns
        raise ValueError("Mahdi Mashayekhi dataset must contain 'title' and 'text' columns.")
    return df[["title", "text", "label"]]

def load_combined_dataset(data_dir: str = "data", extra_path: str = "data/Mahdi Mashayekhi/fake_news_dataset.csv"):
    """Load ISOT and Mahdi Mashayekhi dataset, balance ISOT first, then combine."""
    df_isot = load_isot_dataset(data_dir)
    df_extra = load_additional_dataset(extra_path)

    # Balance ISOT only (since Mahdi dataset is already balanced)
    counts = df_isot['label'].value_counts()
    min_count = counts.min()
    df_isot_balanced = (
        df_isot.groupby('label', group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    combined = pd.concat([df_isot_balanced, df_extra], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Loaded combined dataset: {len(combined)} rows "
          f"(real={combined['label'].value_counts().get(0,0)}, "
          f"fake={combined['label'].value_counts().get(1,0)})")

    return combined


def fetch_current_news(api_key=None, query="news", language="en", page_size=50) -> List[Dict]:
    if api_key is None:
        api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Missing NEWSAPI_KEY. Set environment variable or .env file.")
    params = {"q": query, "language": language, "pageSize": page_size, "sortBy": "publishedAt"}
    headers = {"Authorization": api_key}
    resp = requests.get(NEWSAPI_ENDPOINT, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    normalized = []
    for a in articles:
        normalized.append({
            "title": a.get("title", "") or "",
            "description": a.get("description", "") or "",
            "content": a.get("content", "") or "",
            "url": a.get("url", "") or "",
            "source": a.get("source", {}).get("name", "") or ""
        })
    return normalized