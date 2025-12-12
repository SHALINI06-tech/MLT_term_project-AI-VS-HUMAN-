# src/compute_signature.py
import math, joblib, numpy as np, re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# load pipelines/models (assumes models exist)
nb = joblib.load("models/baselines/nb.joblib")
svm = joblib.load("models/baselines/svm.joblib")
cls = pipeline("text-classification", model="models/distilbert", tokenizer="models/distilbert", return_all_scores=False)

# optional GPT-2 for perplexity-like score
lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm = AutoModelForCausalLM.from_pretrained("gpt2")

def rep_rate(text, n=3):
    toks = text.split()
    ngrams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    if not ngrams: return 0.0
    return 1 - len(set(ngrams))/len(ngrams)

def punct_ratio(text):
    punct = sum(1 for c in text if c in ".,;:!?")
    return punct/len(text) if len(text)>0 else 0

def gpt2_score(text):
    inputs = lm_tokenizer(text, return_tensors="pt")
    with __import__("torch").no_grad():
        logits = lm(**inputs).logits
    # compute pseudo-perplexity (sum negative log-prob / tokens)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    log_probs = __import__("torch").nn.functional.log_softmax(shift_logits, dim=-1)
    tok_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    ppl = float(__import__("torch").exp(-tok_log_probs.mean()))
    return ppl

def signature_score(text):
    # classifier probs
    out = cls(text)[0]
    # HF pipeline labels might be 'LABEL_0' etc; assume label 'LABEL_1' ~ AI
    p_ai = out["score"] if "LABEL_1" in out["label"] or out["label"]=="LABEL_1" else (1-out["score"])
    # baseline prob via NB
    nb_p = nb.predict_proba([text])[0][1]
    sv_p = None
    try:
        sv = svm.decision_function([text])[0]
        # map decision to prob signature via logistic
        sv_p = 1/(1+math.exp(-sv/1.5))
    except:
        sv_p = nb_p
    # stylometric
    rr = rep_rate(text)
    pr = punct_ratio(text)
    # LM perplexity (higher ppl -> less fluent -> perhaps human), normalize
    try:
        ppl = gpt2_score(text)
        lm_score = 1/(1+math.log1p(ppl))  # smaller ppl -> higher score
    except Exception as e:
        lm_score = 0.5
    # entropy of distilbert logits approximation
    import math
    ent = - (p_ai*math.log(max(p_ai,1e-9)) + (1-p_ai)*math.log(max(1-p_ai,1e-9)))
    ent_score = 1 - ent/ math.log(2)  # normalize roughly into 0..1
    # combine with weights
    score = (0.4 * p_ai) + (0.15 * nb_p) + (0.15 * sv_p) + (0.15 * lm_score) + (0.075 * rr) + (0.05 * pr) + (0.05 * ent_score)
    score = max(0.0, min(1.0, score))
    return {
        "signature_score": score,
        "components": {"distilbert": p_ai, "nb": nb_p, "svm": sv_p, "lm_score": lm_score, "repetition": rr, "punct_ratio": pr, "entropy_score": ent_score}
    }
