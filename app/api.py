# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.compute_signature import signature_score

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict(inp: TextIn):
    res = signature_score(inp.text)
    verdict = "AI" if res["signature_score"] > 0.5 else "Human"
    return {"verdict": verdict, "score": res["signature_score"], "components": res["components"]}
