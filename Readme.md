# 📰 Fake News Detection System  
### Comparative Study: TF-IDF Logistic Regression vs DistilBERT Transformer

---

## 🚀 Overview

This project implements a large-scale **Fake News Detection system** using two fundamentally different machine learning paradigms:
- **Classical NLP Model:** TF-IDF + Logistic Regression  
- **Transformer Model:** Fine-tuned DistilBERT  

Rather than focusing only on accuracy, this project evaluates:
- Model behavior differences  
- Recall bias trade-offs  
- Operational stability  
- Deployment feasibility  
- Production suitability  

The system was trained and evaluated on a combined dataset of **62,834 labeled news articles**.

---

## 🎯 Problem Motivation

Misinformation detection is critical in:
- Social media moderation systems  
- News aggregation platforms  
- Search engine ranking pipelines  
- Content trust scoring systems  

In real-world systems, the key challenge is:
> Should we maximize fake-news detection (high recall)  
> or avoid falsely flagging legitimate journalism?

This project explores that trade-off through empirical comparison.

---

## 📚 Dataset

**Total Samples:** 62,834  
**Classes:** Real vs Fake (Balanced)

### Sources
- ISOT Fake News Dataset (2016–2017)
- Mahdi Mashayekhi Fake News Dataset (2025)

### Dataset Strategy

The datasets were combined to:
- Reduce source-specific bias  
- Improve generalization  
- Simulate domain drift across time  
- Increase diversity in writing styles  

⚠️ Raw datasets are not tracked in this repository.  
See the **Data Setup** section below.

---

## 🧹 Data Engineering Pipeline

Implemented in `src/preprocess.py`.

### Preprocessing Steps
- Lowercasing
- URL removal
- HTML tag stripping
- Punctuation removal
- Non-alphabetic token filtering
- Stopword removal (NLTK)
- Lemmatization
- Title + article body concatenation

The preprocessing pipeline is modular and reusable across both models.

---

## 🧠 Modeling Approaches

---

### 1️⃣ TF-IDF + Logistic Regression

**Configuration**
- 30,000 TF-IDF features  
- Uni-grams + Bi-grams  
- L2 regularization  
- Balanced class weighting  

**Strengths**
- Interpretable coefficients  
- Fast training and inference  
- Low memory footprint  
- Cost-efficient deployment  
- Stable under domain shifts  

---

### 2️⃣ DistilBERT Transformer

**Configuration**
- Pretrained DistilBERT encoder (HuggingFace)  
- Max token length: 256  
- AdamW optimizer  
- Early stopping  
- Decision threshold tuning  

**Strengths**
- Context-aware embeddings  
- Captures semantic relationships  
- Strong performance on NLP tasks  
- Learns nuanced deceptive patterns  

---

## 📊 Evaluation Metrics

Both models evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

This ensures performance analysis beyond simple accuracy.

---

## 📈 Results Summary

### 🔹 TF-IDF + Logistic Regression

| Metric | Score |
|--------|-------|
| Accuracy | 83.64% |
| Precision | Balanced |
| Recall | Balanced |
| ROC-AUC | Strong separation |

**Behavior:** Stable, balanced classification across both classes.

---

### 🔹 DistilBERT Transformer

| Metric | Score |
|--------|-------|
| Accuracy | 83.71% |
| Fake Recall | 100% |
| Real Recall | 67% |
| ROC-AUC | Comparable |

**Behavior:** Highly sensitive to deceptive cues.  
Tends to over-predict the Fake class.

---

## 🔍 Key Insight

Despite nearly identical accuracy:
- The transformer aggressively prioritizes detecting fake articles.
- The classical model provides more balanced predictions.

This demonstrates a critical production lesson:
> Higher model complexity does not automatically guarantee operational superiority.

In real moderation systems, excessive false positives can:
- Suppress legitimate journalism  
- Damage platform trust  
- Increase legal and compliance risk  

---

## ⚙️ Production Considerations

| Factor | TF-IDF Model | DistilBERT |
|--------|--------------|------------|
| Training Time | Fast | High |
| Inference Latency | Low | Moderate |
| GPU Required | No | Recommended |
| Interpretability | High | Low |
| Deployment Cost | Minimal | Higher |
| Scalability | High | Moderate |

---

## 🏗 System Architecture

See diagrams in:

```text
figures/
├── System_Architecture.svg
└── Workflow.svg
```

---

## 📁 Project Structure

```text
.
├── app/
│   └── app.py
├── figures/
│   ├── System_Architecture.svg
│   └── Workflow.svg
├── results/
│   ├── tfidf/
│   │   ├── confusion_matrix.png
│   │   └── roc_auc.png
│   └── transformer/
│       ├── confusion_matrix.png
│       ├── loss_curve.png
│       └── roc_auc.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── evaluate_transformer.py
│   ├── predict.py
│   └── predict_transformer.py
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

### 🏋️ Training

**Train TF-IDF Model**
```bash
python src/train.py
```

**Train Transformer Model**
```bash
python src/train_transformer.py
```

### 🔮 Inference

**Classical Model**
```bash
python src/predict.py --text "Your news article here"
```

**Transformer Model**
```bash
python src/predict_transformer.py --text "Your news article here"
```

---

## 📦 Data Setup

Download datasets from their respective public sources and place them locally under:

```text
data/
```
*Raw datasets and trained model artifacts are intentionally excluded from version control.*

---

## 🧪 Technical Stack

- Python
- Scikit-learn
- PyTorch
- HuggingFace Transformers
- NLTK
- Pandas
- NumPy

---

## 🚧 Limitations

- Dataset bias from public sources
- No multilingual support
- No adversarial robustness testing
- No real-time streaming deployment
- Limited calibration analysis

---

## 🔮 Future Improvements

- Ensemble modeling
- Model calibration tuning
- Drift detection monitoring
- Adversarial training
- FastAPI deployment
- Docker containerization
- CI/CD integration

---

## 👤 Author

**Srikanth Reddy Nandireddy** M.S. Data Science & Artificial Intelligence  
University of Central Missouri  

**Interests:**
- Applied Machine Learning  
- NLP Systems  
- Production ML Engineering  
- Model Evaluation & Optimization  

---

## 📜 License

Released for academic and research purposes.