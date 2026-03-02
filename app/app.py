import sys, os, time, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from dotenv import load_dotenv

# local predictors
from src.predict import load_model as load_tfidf_model, predict_text as predict_tfidf_text, predict_from_article_list as predict_tfidf_articles
from src.predict_transformer import load_transformer, predict_text as predict_transformer_text, predict_from_article_list as predict_transformer_articles, get_current_threshold
from src.data_utils import fetch_current_news

load_dotenv()

# ------------------------------------------------------------------
# Paths (match the training output structure from the scripts)
# ------------------------------------------------------------------
TFIDF_MODEL_PATH = "models/tfidf/model.pkl"
TRANSFORMER_MODEL_DIR = "models/transformer"
RESULTS_TFIDF = "results/tfidf"
RESULTS_TRANSFORMER = "results/transformer"

# ------------------------------------------------------------------
# Page config & style (kept concise; you can paste full CSS from earlier)
# ------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection — Premium UI", layout="wide")
st.title("📰 Fake News Detection — TF-IDF vs DistilBERT")

# small helper load functions
@st.cache_resource
def _load_tfidf_cached():
    try:
        return load_tfidf_model(TFIDF_MODEL_PATH)
    except Exception as e:
        st.error(f"TF-IDF model load error: {e}")
        return None

@st.cache_resource
def _load_transformer_cached():
    try:
        return load_transformer(TRANSFORMER_MODEL_DIR)
    except Exception as e:
        st.error(f"Transformer load error: {e}")
        return None

def read_text_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception:
        return None

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Analyze Text", "Live News", "Compare Models", "About / Results"])

with tab1:
    st.header("Analyze Text")
    text = st.text_area("Paste headline or article text", height=240)

    col1, col2 = st.columns([2,1])
    with col1:
        model_choice = st.selectbox("Model", ["TF-IDF + Logistic Regression", "DistilBERT Transformer"])
        explain = st.checkbox("Show keyword explanation (heuristic)", value=True)
    with col2:
        analyze_btn = st.button("Analyze")

    if analyze_btn:
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            st.info("Running model(s)...")
            results = []
            if model_choice.startswith("TF-IDF"):
                model = _load_tfidf_cached()
                if model:
                    tf_res = predict_tfidf_text(text, model=model)
                    results.append(("TF-IDF", tf_res))
            else:
                loaded = _load_transformer_cached()
                if loaded:
                    model_tf, tokenizer, device = loaded
                    tr_res = predict_transformer_text(text, model=model_tf, tokenizer=tokenizer, device=device)
                    # apply threshold if available
                    threshold = get_current_threshold(TRANSFORMER_MODEL_DIR)
                    label = 1 if tr_res.get("probability_fake", 0.0) > threshold else 0
                    tr_res["threshold_applied"] = threshold
                    tr_res["prediction_label"] = int(label)
                    results.append(("DistilBERT", tr_res))

            # display
            for name, r in results:
                st.markdown(f"### {name} result")
                if name == "TF-IDF":
                    lab = "🚨 Fake" if r["prediction"] == 1 else "✅ Real"
                    st.metric("Label", lab)
                    st.progress(r["probability"])
                    st.caption(f"Probability (Fake): {r['probability']:.3f}")
                else:
                    if r["prediction"] == -1:
                        st.info("Input too short for transformer classification.")
                    else:
                        lab = "🚨 Fake" if r.get("prediction_label", r.get("prediction")) == 1 else "✅ Real"
                        st.metric("Label", lab)
                        st.progress(r.get("probability_fake", 0.0))
                        st.caption(f"Probability (Fake): {r.get('probability_fake', 0.0):.3f} (Threshold: {r.get('threshold_applied', 0.60):.2f})")

            if explain:
                st.markdown("#### Keyword explanation (heuristic)")
                # pseudo-explain using TF-IDF feature importances if available
                # try reading results/tfidf/feature_importances.json
                fi_path = os.path.join(RESULTS_TFIDF, "feature_importances.json")
                fi = None
                try:
                    if os.path.exists(fi_path):
                        import json
                        with open(fi_path, 'r') as f:
                            fi = json.load(f)
                except Exception:
                    fi = None

                # fallback simple highlight: split important tokens from model outputs
                tokens = [t.strip() for t in text.split() if len(t) > 3][:40]
                # show tokens simply (UI placeholder for the more advanced pseudo-shap we had earlier)
                st.write("Important tokens (example):")
                st.write(", ".join(tokens[:20]))

with tab2:
    st.header("Live News (NewsAPI)")
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        st.info("Set NEWSAPI_KEY in your .env to enable live fetch.")
    else:
        q = st.text_input("Query", value="technology OR politics")
        n = st.slider("Number of articles", 1, 20, 8)
        model_choice_live = st.selectbox("Model for live fetch", ["TF-IDF + Logistic Regression", "DistilBERT Transformer"], key="live_model")
        if st.button("Fetch & Classify"):
            try:
                arts = fetch_current_news(api_key=api_key, query=q, page_size=n)
                if not arts:
                    st.info("No articles returned.")
                else:
                    if model_choice_live.startswith("TF-IDF"):
                        model = _load_tfidf_cached()
                        res = predict_tfidf_articles(arts, model=model)
                    else:
                        loaded = _load_transformer_cached()
                        model, tokenizer, device = loaded
                        res = predict_transformer_articles(arts, model=model, tokenizer=tokenizer, device=device)
                    st.write("### Results:")
                    for r in res:
                        if r.get("prediction", -1) == -1:
                            st.write(f"**{r['title']}** — skipped (too short)")
                            continue
                        label = "🚨 Fake" if r.get("prediction", 0) == 1 else "✅ Real"
                        st.markdown(f"**{r['title']}** — *{r.get('source','')}* — {label} ({r.get('probability', r.get('probability_fake',0.0)):.2f})")
                        if r.get("url"):
                            st.markdown(f"[Open]({r.get('url')})")
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.header("Compare Models")
    compare_text = st.text_area("Paste text to compare both models", height=200)
    if st.button("Compare"):
        if not compare_text.strip():
            st.warning("Enter text")
        else:
            # TF-IDF
            model = _load_tfidf_cached()
            tf_res = None
            if model:
                tf_res = predict_tfidf_text(compare_text, model=model)
            # Transformer
            loaded = _load_transformer_cached()
            tr_res = None
            if loaded:
                m,tok,dev = loaded
                tr_res = predict_transformer_text(compare_text, model=m, tokenizer=tok, device=dev)
                th = get_current_threshold(TRANSFORMER_MODEL_DIR)
                tr_res["prediction_label"] = 1 if tr_res.get("probability_fake",0.0) > th else 0
                tr_res["threshold"] = th
            # Show side-by-side
            cols = st.columns(2)
            with cols[0]:
                st.subheader("TF-IDF")
                if tf_res:
                    st.write("Prediction:", "Fake" if tf_res["prediction"] == 1 else "Real")
                    st.progress(tf_res["probability"])
            with cols[1]:
                st.subheader("DistilBERT")
                if tr_res:
                    st.write("Prediction:", "Fake" if tr_res.get("prediction_label", tr_res.get("prediction")) == 1 else "Real")
                    st.progress(tr_res.get("probability_fake", 0.0))
            st.write("You can use the About tab to inspect saved metrics and plots.")

with tab4:
    st.header("About & Results")
    st.markdown("Model files and training results are expected at:")
    st.code("""
        models/tfidf/model.pkl
        models/transformer/   (HuggingFace model + best_threshold.txt)
        results/tfidf/
        results/transformer/
    """)
    st.markdown("### TF-IDF results (if available)")
    tf_metrics = read_text_file(os.path.join(RESULTS_TFIDF, "performance_metrics.txt"))
    if tf_metrics:
        st.code(tf_metrics)
    else:
        st.info("No TF-IDF metrics found in results/tfidf")

    st.markdown("### Transformer results (if available)")
    tr_metrics = read_text_file(os.path.join(RESULTS_TRANSFORMER, "performance_metrics.txt"))
    if tr_metrics:
        st.code(tr_metrics)
    else:
        st.info("No Transformer metrics found in results/transformer")

    st.markdown("### Plots")
    if os.path.exists(os.path.join(RESULTS_TFIDF, "roc_auc.png")):
        st.image(os.path.join(RESULTS_TFIDF, "roc_auc.png"), caption="TF-IDF ROC")
    if os.path.exists(os.path.join(RESULTS_TRANSFORMER, "roc_auc.png")):
        st.image(os.path.join(RESULTS_TRANSFORMER, "roc_auc.png"), caption="Transformer ROC")
    if os.path.exists(os.path.join(RESULTS_TRANSFORMER, "loss_curve.png")):
        st.image(os.path.join(RESULTS_TRANSFORMER, "loss_curve.png"), caption="Transformer Loss Curve")

st.caption("⚠️ Predictions are experimental. Use for research/demo only.")