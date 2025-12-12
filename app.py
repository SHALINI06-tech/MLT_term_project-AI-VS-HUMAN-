import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="model",
    tokenizer="model",
    return_all_scores=True
)

def detect_ai(text):
    preds = classifier(text)[0]
    ai_score = preds[1]["score"]
    human_score = preds[0]["score"]

    if ai_score > human_score:
        label = "AI-Generated"
    else:
        label = "Human-Written"

    return {
        "Prediction": label,
        "AI Probability": round(ai_score, 3),
        "Human Probability": round(human_score, 3),
    }

ui = gr.Interface(
    fn=detect_ai,
    inputs=gr.Textbox(lines=7, label="Enter text"),
    outputs="json",
    title="AI Text Detector (DistilBERT)",
    description="Detect whether text is AI-generated or human-written with a fine-tuned DistilBERT model."
)

ui.launch()
