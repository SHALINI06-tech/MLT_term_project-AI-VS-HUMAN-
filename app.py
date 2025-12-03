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

# ----------------- GLASSMORPHISM DARK MODE CSS -----------------
dark_css = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    
    /* Header glassmorphism card */
    .glass-header {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .glass-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.8), transparent);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Input area glassmorphism */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem !important;
        padding: 1.2rem !important;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 1px solid rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 8px 24px 0 rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
    }
    
    .stTextArea > label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 8px 24px 0 rgba(99, 102, 241, 0.3) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        letter-spacing: 0.02em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px 0 rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Result card glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-badge {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-badge.ai {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .result-badge.human {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .result-label.ai {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .result-label.human {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Probability section */
    .prob-section {
        margin-top: 1.5rem;
    }
    
    .prob-title {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .prob-item {
        margin-bottom: 1.2rem;
    }
    
    .prob-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .prob-bar {
        height: 12px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prob-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .prob-fill.human {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.4);
    }
    
    .prob-fill.ai {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }
    
    /* Warning message */
    .stAlert {
        background: rgba(251, 191, 36, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: 12px !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(dark_css, unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
model = joblib.load("ai_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ----------------- HEADER -----------------
st.markdown("""
<div class="glass-header">
    <div class="main-title">🤖 AI vs Human Text Detector</div>
    <div class="subtitle">Advanced machine learning analysis to detect AI-generated content</div>
</div>
""", unsafe_allow_html=True)

# ----------------- INPUT AREA -----------------
user_input = st.text_area("✍️ Enter text to analyze:", height=200, placeholder="Paste or type your text here...")

if st.button("🔍 Analyze Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Process the input
        X = vectorizer.transform([user_input])
        prob = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        
        ai_prob = float(prob[1])
        human_prob = float(prob[0])
        
        # Display results
        badge_class = "ai" if pred == 1 else "human"
        label_text = "⚠️ AI-Generated Text" if pred == 1 else "✅ Human-Written Text"
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-badge {badge_class}">
                <div class="result-label {badge_class}">{label_text}</div>
            </div>
            
            <div class="prob-section">
                <div class="prob-title">📊 Confidence Analysis</div>
                
                <div class="prob-item">
                    <div class="prob-label">
                        <span>👤 Human</span>
                        <span style="color: #22c55e; font-weight: 700;">{human_prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill human" style="width: {human_prob*100}%;"></div>
                    </div>
                </div>
                
                <div class="prob-item">
                    <div class="prob-label">
                        <span>🤖 AI</span>
                        <span style="color: #ef4444; font-weight: 700;">{ai_prob*100:.1f}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill ai" style="width: {ai_prob*100}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
