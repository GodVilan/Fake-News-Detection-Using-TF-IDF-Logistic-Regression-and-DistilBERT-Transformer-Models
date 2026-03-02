# 📰 Fake News Detection System: Classical NLP vs. Transformers
### An End-to-End Machine Learning Pipeline & Comparative Study

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-FFD21E?style=for-the-badge&color=FFD21E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

</div>
---

## 🚀 Overview

This repository features an end-to-end Machine Learning application designed to detect misinformation. Beyond a simple comparative study, this project implements a full **production-style pipeline**—from data ingestion and model training to inference and live deployment via a Streamlit web interface. 

The system compares two fundamentally different machine learning paradigms:
1. **Classical NLP:** TF-IDF + Logistic Regression (Optimized for speed, interpretability, and low footprint).
2. **Deep Learning:** Fine-tuned `distilbert-base-uncased` (Optimized for contextual semantic understanding).

Rather than chasing raw accuracy, this project evaluates operational stability, decision-threshold tuning for recall biases, feature explainability, and deployment feasibility.

---

## ✨ Key Engineering Features

- **Live News Integration:** Integrates with the `NewsAPI` to fetch real-time articles and classify them dynamically in the UI.
- **Dynamic Threshold Tuning:** The DistilBERT model doesn't just output argmax probabilities; it calculates the optimal decision threshold based on maximizing the F1-score during validation to handle recall trade-offs.
- **Feature Explainability:** The TF-IDF pipeline extracts and stores feature importances (top predictive keywords for Real vs. Fake) to provide heuristic explainability in the UI.
- **Interactive UI:** A multi-tab Streamlit dashboard allowing users to input text, fetch live news, and compare both models side-by-side.
- **Automated Experiment Tracking:** Training scripts automatically log classification reports, confusion matrices, ROC-AUC curves, and loss curves to a standardized `results/` directory.

---

## 📚 Data Engineering

**Total Samples:** 62,834 | **Classes:** Real vs Fake (Balanced)

* **ISOT Fake News Dataset** (2016–2017)
* **Mahdi Mashayekhi Fake News Dataset** (2025)

**Pipeline (`src/preprocess.py` & `src/data_utils.py`):**
- Automated balancing of source datasets prior to concatenation.
- NLTK-based cleaning pipeline: URL and HTML stripping, non-alphabetic filtering, stopword removal, and WordNet lemmatization.

*⚠️ Note: Raw datasets are excluded from version control. See the Data Setup section below.*

---

## 🧠 Model Architectures & Performance

### 1️⃣ TF-IDF + Logistic Regression
- **Architecture:** 30,000 max features, Uni-grams & Bi-grams, `saga` solver, balanced class weighting.
- **Performance Focus:** Highly interpretable. Achieves strong baseline separation with minimal inference latency. Excellent for cost-efficient deployment.
- **Explainability:** Feature weights are serialized to JSON, allowing the UI to highlight tokens driving the "Fake" or "Real" predictions.

### 2️⃣ Fine-Tuned DistilBERT
- **Architecture:** `distilbert-base-uncased` fine-tuned with HuggingFace `Trainer`, AdamW optimizer, and dynamic padding.
- **Performance Focus:** Captures nuanced, deceptive semantic patterns. Includes an automated script that sweeps probability thresholds (0.1 to 0.9) to find the exact cut-off that maximizes the F1-score, reducing false positives against legitimate journalism.

---

## 🏗 System Architecture & Repository Structure

```text
.
├── app.py                         # Streamlit Web Application
├── data/                          # Raw datasets (Not tracked)
├── models/                        # Serialized models (.pkl & HF weights)
├── results/                       # Auto-generated artifacts
│   ├── tfidf/
│   │   ├── classification_report.txt (Not tracked)
│   │   ├── confusion_matrix.png
│   │   ├── feature_importances.json (Not tracked)
│   │   ├── metadata.json (Not tracked)
│   │   ├── performance_metrics.txt (Not tracked)
│   │   └── roc_auc.png
│   └── transformer/
│       ├── classification_report.txt (Not tracked)
│       ├── confusion_matrix.png
│       ├── loss_curve.png
│       ├── metadata.json (Not tracked)
│       ├── performance_metrics.txt (Not tracked)
│       ├── roc_auc.png
│       └── threshold.txt          # Dynamically tuned F1 threshold
├── src/                           # Backend Pipeline Modules
│   ├── data_utils.py              # Data ingestion & NewsAPI logic
│   ├── evaluate.py                # TF-IDF evaluation logic
│   ├── evaluate_transformer.py    # DistilBERT evaluation & threshold tuning
│   ├── predict.py                 # Classical inference wrapper
│   ├── predict_transformer.py     # Deep learning inference wrapper
│   ├── preprocess.py              # NLTK text cleaning pipeline
│   ├── train.py                   # TF-IDF training script
│   └── train_transformer.py       # DistilBERT fine-tuning script
├── requirements.txt
└── README.md
```

---

## 🛠 Local Setup & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/GodVilan/Fake-News-Detection-Using-TF-IDF-Logistic-Regression-and-DistilBERT-Transformer-Models.git](https://github.com/GodVilan/Fake-News-Detection-Using-TF-IDF-Logistic-Regression-and-DistilBERT-Transformer-Models.git)
cd Fake-News-Detection-Using-TF-IDF-Logistic-Regression-and-DistilBERT-Transformer-Models

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Download the ISOT and Mahdi Mashayekhi datasets and place them in the `data/` folder:
- `data/ISOT/True.csv` and `data/ISOT/Fake.csv`
- `data/Mahdi Mashayekhi/fake_news_dataset.csv`

### 3. Environment Variables
To enable the Live News fetching feature in the UI, create a `.env` file in the root directory:
```env
NEWSAPI_KEY=your_api_key_here
```

---

## 🏋️ Training the Models

Run the training scripts from the root directory. Results, plots, and metadata will be automatically saved to the `results/` folder, and models to the `models/` folder.

**Train TF-IDF Pipeline:**
```bash
python -m src.train
```

**Fine-tune DistilBERT (GPU Recommended):**
```bash
python -m src.train_transformer --epochs 3 --batch_size 16
```

---

## 💻 Running the Streamlit App

Launch the interactive dashboard to analyze text, compare models, and fetch live news:

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements
- **Ensemble Inference:** Combining the TF-IDF feature weights with DistilBERT embeddings.
- **Drift Detection:** Implementing monitoring for data drift as news writing styles evolve.
- **Dockerization:** Containerizing the Streamlit app and FastAPI backend for scalable cloud deployment.

---

## 👤 Author

**Srikanth Reddy Nandireddy** *M.S. Data Science & Artificial Intelligence* *University of Central Missouri* **Focus Areas:** Applied Machine Learning | NLP Systems | Production ML Engineering | Generative AI

---
*Released for academic, portfolio, and research purposes.*