"""
Usage:
  python -m src.evaluate --model models/odel.pkl --data_dir data
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from src.data_utils import load_isot_dataset
from src.preprocess import concat_and_clean

def evaluate(model_path: str, data_dir: str):
    model = joblib.load(model_path)
    df = load_isot_dataset(data_dir)
    df['text_all'] = df.apply(lambda r: concat_and_clean(r['title'], r['text']), axis=1)
    X = df['text_all']
    y = df['label']
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/tfidf/model.pkl')
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    evaluate(args.model, args.data_dir)
