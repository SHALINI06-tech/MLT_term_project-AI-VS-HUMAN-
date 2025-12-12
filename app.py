import streamlit as st
import joblib

# ===========================
# Load Model & Vectorizer
# ===========================
model = joblib.load("ai_detector_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ===========================
# DARK THEME CSS
# ===========================
dark_css = """
<style>
body {
    background-color: #0E1117;
    color: white;
}
textarea, input {
    background-color: #1A1D23 !important;
    color: white !important;
}
.stTextArea textarea {
    background-color: #1E222A !important;
    color: white !important;
}
</style>
"""

st.markdown(dark_css, unsafe_allow_html=True)

# ===========================
# App UI
# ===========================
st.title("🔥 AI vs Human Text Detector")

st.write("Enter some text and the model will classify it as AI-generated or Human-written.")

user_text = st.text_area("Enter your text here:")

if st.button("Analyze Text"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform input
        X_input = tfidf.transform([user_text])

        # Predict
        pred = model.predict(X_input)[0]

        label = "🤖 AI-Generated Text" if pred == 1 else "🧑 Human-Written Text"

        st.success(f"Prediction: **{label}**")
