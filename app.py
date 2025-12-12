import streamlit as st
import joblib
import numpy as np

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI vs Human Detector",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------- LOAD MODEL WITH SAFETY -----------------
@st.cache_resource
def load_model_safe():
    try:
        model = joblib.load("ai_detector_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, load_error = load_model_safe()

# ----------------- CUSTOM CSS -----------------
# (Paste your dark_css code here, unchanged)
st.markdown(dark_css, unsafe_allow_html=True)

# ----------------- HEADER -----------------
st.markdown("""
<div class="glass-header">
    <div class="main-title">🤖 AI vs Human Text Detector</div>
    <div class="subtitle">Machine-learning based estimation — not 100% accurate</div>
</div>
""", unsafe_allow_html=True)

# ----------------- ERROR IF MODEL FAILED -----------------
if load_error:
    st.error("❌ Model could not be loaded. Please check your files.\n\n"
             f"**Error:** {load_error}")
    st.stop()

# ----------------- OPTIONAL PREPROCESSING -----------------
def clean_text(x):
    import re
    x = x.lower()
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

use_preprocessing = st.checkbox("🧹 Apply text preprocessing", value=False)

# ----------------- INPUT -----------------
user_input = st.text_area(
    "✍️ Enter text to analyze:",
    height=200,
    placeholder="Paste or type your text here..."
)

# ----------------- BUTTON -----------------
if st.button("🔍 Analyze Text"):

    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
        st.stop()

    text = clean_text(user_input) if use_preprocessing else user_input

    # Predict
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    ai_prob = float(prob[1])
    human_prob = float(prob[0])

    confidence = max(ai_prob, human_prob)
    if confidence > 0.85:
        confidence_level = "🔵 High Confidence"
    elif confidence > 0.65:
        confidence_level = "🟡 Medium Confidence"
    else:
        confidence_level = "🟠 Low Confidence"

    # Display results
    badge_class = "ai" if pred == 1 else "human"
    label_text = "⚠️ AI-Generated Text" if pred == 1 else "✅ Human-Written Text"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-badge {badge_class}">
            <div class="result-label {badge_class}">{label_text}</div>
        </div>

        <div style="text-align:center; margin-top:-10px;">
            <span style="font-size:1rem; opacity:0.85;">{confidence_level}</span>
        </div>

        <div class="prob-section">
            <div class="prob-title">📊 Confidence Breakdown</div>

            <div class="prob-item">
                <div class="prob-label">
                    <span>👤 Human</span>
                    <span style="color: #22c55e; font-weight:700;">{human_prob*100:.1f}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill human" style="width: {human_prob*100}%;"></div>
                </div>
            </div>

            <div class="prob-item">
                <div class="prob-label">
                    <span>🤖 AI</span>
                    <span style="color: #ef4444; font-weight:700;">{ai_prob*100:.1f}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill ai" style="width: {ai_prob*100}%;"></div>
                </div>
            </div>
        </div>

        <div style="margin-top:20px; font-size:0.9rem; color:rgba(255,255,255,0.6); text-align:center;">
            ⚠️ <b>Note:</b> This model is not perfect. It may misclassify well-written human text or AI-generated text.
            Use results as guidance, not absolute truth.
        </div>
    </div>
    """, unsafe_allow_html=True)
