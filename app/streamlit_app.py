# app/streamlit_app.py
import streamlit as st
import joblib
from transformers import pipeline
from src.compute_signature import signature_score  # adjust import path as needed
import pandas as pd

st.set_page_config(page_title="AI vs Human Detector", layout="centered")
st.title("AI vs Human Detector — DistilBERT + Ensemble")
st.write("Paste text below to analyze.")

# load models
@st.cache_resource
def load_models():
    distil = pipeline("text-classification", model="models/distilbert", tokenizer="models/distilbert", device=0 if __import__("torch").cuda.is_available() else -1)
    nb = joblib.load("models/baselines/nb.joblib")
    svm = joblib.load("models/baselines/svm.joblib")
    return distil, nb, svm

distil, nb, svm = load_models()

text = st.text_area("Enter text:", height=300)
if st.button("Analyze"):
    if not text.strip():
        st.warning("Enter text")
    else:
        with st.spinner("Predicting..."):
            d_out = distil(text)[0]
            # compute probs
            # map label
            ai_prob = d_out["score"] if d_out["label"] in ("LABEL_1","AI","label_1") else 1-d_out["score"]
            nb_prob = nb.predict_proba([text])[0][1]
            try:
                sv = svm.decision_function([text])[0]
                sv_prob = 1/(1+__import__("math").exp(-sv/1.5))
            except:
                sv_prob = nb_prob
            sig = signature_score(text)

        st.metric("DistilBERT AI prob", f"{ai_prob:.3f}")
        st.metric("NB AI prob", f"{nb_prob:.3f}")
        st.metric("SVM AI prob", f"{sv_prob:.3f}")
        st.subheader("AI Writing Signature Score")
        st.progress(int(sig["signature_score"]*100))
        st.json(sig["components"])
        st.write("Final verdict:", "🤖 AI-Generated" if sig["signature_score"]>0.5 else "🧑 Human-Written")
