import streamlit as st
from transformers import pipeline
import torch

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="AI vs Human Text Detector",
    page_icon="🤖",
    layout="centered",
)

# ----------------------------------
# CUSTOM DARK THEME CSS
# ----------------------------------
dark_css = """
<style>
body {
    background-color: #0E1117;
    color: white;
}

textarea, input {
    background-color: #1E1E1E !important;
    color: white !important;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 0.6rem;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ----------------------------------
# HEADER
# ----------------------------------
st.title("🤖 AI vs Human Text Detector")
st.write("Detect whether the text was written by **AI or Human** using a fine-tuned DistilBERT model.")

# ----------------------------------
# LOAD MODEL PIPELINE
# ----------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="./ai_detector_model",
        tokenizer="./ai_detector_model",
        device=0 if torch.cuda.is_available() else -1
    )

classifier = load_model()

# ----------------------------------
# INPUT BOX
# ----------------------------------
text = st.text_area("Enter text to analyze:", height=200)

if st.button("Analyze Text"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)

        is_ai = label == "LABEL_1"

        if is_ai:
            st.markdown(
                f"<div class='result-box' style='background-color:#8B0000;'>"
                f"<h3>🤖 Prediction: AI-Generated Text</h3>"
                f"<p>Confidence: <b>{score}%</b></p>"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box' style='background-color:#004d00;'>"
                f"<h3>🧑 Human-Written Text</h3>"
                f"<p>Confidence: <b>{score}%</b></p>"
                "</div>", unsafe_allow_html=True
            )
