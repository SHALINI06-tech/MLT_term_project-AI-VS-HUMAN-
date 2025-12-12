import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖", layout="centered")

# CSS (small)
st.markdown("""
<style>
body { background-color: #0f1724; color: #e6eef5; }
.stButton>button { background: linear-gradient(90deg,#6366f1,#8b5cf6); color: white; border-radius: 8px; padding: 8px 20px; }
textarea { background-color: #0b1220; color: #e6eef5; border-radius:8px; padding:10px; }
.result { padding:16px; border-radius:10px; margin-top:12px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI vs Human Text Detector")
st.write("Fine-tuned DistilBERT — probability + label")

MODEL_DIR = "./model"

@st.cache_resource
def load_pipeline(model_dir=MODEL_DIR):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model_dir, tokenizer=model_dir, device=device, return_all_scores=False)
    return pipe

pipe = load_pipeline()

text = st.text_area("Enter text to analyze:", height=240)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            out = pipe(text)
            # pipeline returns [{'label':'LABEL_0' or 'LABEL_1', 'score':0.xx}]
            label = out[0]["label"]
            score = out[0]["score"]
            # By default HF labels for binaries are LABEL_0 / LABEL_1; check mapping
            # We'll assume LABEL_1 corresponds to class 1 (AI)
            is_ai = (label == "LABEL_1")
            pct = round(score * 100, 2)

        if is_ai:
            st.markdown(f"<div class='result' style='background:#2b0a0a; color:#ffd6d6;'><h3>🤖 AI-Generated</h3><p>Confidence: <b>{pct}%</b></p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result' style='background:#07220a; color:#bfffd0;'><h3>🧑 Human-Written</h3><p>Confidence: <b>{pct}%</b></p></div>", unsafe_allow_html=True)
