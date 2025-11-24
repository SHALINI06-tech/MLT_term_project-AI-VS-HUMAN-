import streamlit as st
import joblib

st.set_page_config(page_title="AI Text Detector", page_icon="🤖", layout="centered")

# Dark UI
st.markdown("""
<style>
body {
    background-color: #0e0e0e;
    color: #ffffff;
}
.stTextArea textarea {
    background-color: #1e1e1e;
    color: #ffffff;
    border-radius: 12px;
    padding: 12px;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
}
.ai-text {
    background-color: #2b005c;
    color: #d7b0ff;
    border: 1px solid #6b00ff;
}
.human-text {
    background-color: #003314;
    color: #7affc9;
    border: 1px solid #00cc88;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Text Detector")
st.write("Detect whether the text is AI-generated or Human-written")

# Load model & vectorizer
model = joblib.load("ai_detector_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

text = st.text_area("Enter your text here...")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please type something!")
    else:
        X = tfidf.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        ai_score = round(proba[1] * 100, 2)
        human_score = round(proba[0] * 100, 2)

        if prediction == 1:
            st.markdown(
                f'<div class="result-box ai-text">🤖 <b>AI-Generated</b><br>Confidence: {ai_score}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box human-text">🧍 <b>Human-Written</b><br>Confidence: {human_score}%</div>',
                unsafe_allow_html=True,
            )
