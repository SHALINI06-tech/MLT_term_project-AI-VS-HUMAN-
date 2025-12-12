#!/usr/bin/env python3
"""
create_dataset.py

Creates a balanced AI vs Human CSV dataset.
Usage:
    python create_dataset.py --size 5000
Outputs:
    dataset.csv  (columns: text,label) where label is 0=human,1=ai
"""

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"<.*?>", "", s)
    return s

def try_load_hf_datasets(target_per_class: int):
    """
    Try various HF datasets for human and AI text. Returns (human_texts, ai_texts)
    May return fewer than requested—caller handles fallback.
    """
    human_texts = []
    ai_texts = []
    try:
        from datasets import load_dataset
    except Exception as e:
        print("datasets library not available:", e)
        return [], []

    # 1) Wikitext for human-ish text (clean)
    try:
        print("Loading wikitext (human source)...")
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
        raw = wikitext["train"]["text"]
        raw = [clean_text(t) for t in raw if isinstance(t, str) and len(t.strip()) > 60]
        human_texts.extend(raw)
        print("Loaded wikitext samples:", len(raw))
    except Exception as e:
        print("wikitext load error:", e)

    # 2) OpenWebText or c4 style human text (try 'openwebtext' if available)
    try:
        print("Loading openwebtext (human source)...")
        openweb = load_dataset("openwebtext", split="train")
        raw = [clean_text(t) for t in openweb["text"] if isinstance(t, str) and len(t.strip()) > 60]
        human_texts.extend(raw)
        print("Loaded openwebtext samples:", len(raw))
    except Exception as e:
        print("openwebtext load error:", e)

    # 3) For AI texts: try 'openai_humaneval' prompts (small) and other known sources
    try:
        print("Loading openai_humaneval (AI source)...")
        humaneval = load_dataset("openai_humaneval", split="test")
        prompts = [clean_text(x["prompt"]) for x in humaneval if x.get("prompt") and len(x["prompt"]) > 40]
        ai_texts.extend(prompts)
        print("Loaded openai_humaneval samples:", len(prompts))
    except Exception as e:
        print("openai_humaneval load error:", e)

    # 4) Try HC3 if possible (some HF scripts blocked)
    try:
        print("Attempting to load HC3 (may fail if script unsupported)...")
        hc3 = load_dataset("Hello-SimpleAI/HC3", "all")
        # HC3 format can have nested lists; handle conservatively
        # extract chatgpt_answers if present
        if "train" in hc3:
            tr = hc3["train"]
            # try common keys
            if "chatgpt_answer" in tr.column_names:
                ai_texts.extend([clean_text(x) for x in tr["chatgpt_answer"] if isinstance(x, str) and len(x) > 40])
            elif "chatgpt_answers" in tr.column_names:
                # list of lists
                flat = []
                for lst in tr["chatgpt_answers"]:
                    if isinstance(lst, list):
                        flat.extend([clean_text(x) for x in lst if isinstance(x, str) and len(x) > 40])
                ai_texts.extend(flat)
        print("Loaded HC3 samples:", len(ai_texts))
    except Exception as e:
        print("HC3 load skipped/error:", e)

    # Remove duplicates, shuffle
    human_texts = list(dict.fromkeys(human_texts))
    ai_texts = list(dict.fromkeys(ai_texts))
    random.shuffle(human_texts)
    random.shuffle(ai_texts)
    return human_texts[:target_per_class], ai_texts[:target_per_class]

def synthetic_ai_from_human(human_texts, needed):
    """
    Create simple synthetic 'AI-like' texts from human texts as a fallback:
    - Expand sentences
    - Add connecting phrases
    - Rephrase with common 'AI' markers like 'In summary', 'Overall', 'As an AI assistant' (but avoid claiming identity)
    This is a fallback and should be replaced with real AI samples when possible.
    """
    out = []
    phrases = [
        "In summary,",
        "Overall,",
        "In conclusion,",
        "To summarize,",
        "Therefore,",
        "Consequently,",
        "It should be noted that",
        "Importantly,"
    ]
    for t in human_texts:
        if len(out) >= needed:
            break
        t = clean_text(t)
        # pick a phrase and prepend/append
        phrase = random.choice(phrases)
        if len(t) < 200:
            candidate = f"{phrase} {t} {random.choice(['Furthermore, this suggests that', 'This demonstrates that', 'This indicates that'])} {t.split('.')[0]}."
        else:
            candidate = f"{phrase} {t}"
        out.append(candidate)
    return out[:needed]

def build_dataset(total_size=5000, out_path="dataset.csv"):
    assert total_size % 2 == 0, "total_size must be even for balanced classes"
    per_class = total_size // 2
    print(f"Target: {per_class} human + {per_class} AI = {total_size}")

    human_texts, ai_texts = try_load_hf_datasets(per_class)

    # fallback: if insufficient human samples, raise error (human sources are plentiful)
    if len(human_texts) < per_class:
        print(f"Found only {len(human_texts)} human samples from HF sources. Will repeat/shuffle to reach target.")
        # replicate by sampling with replacement
        while len(human_texts) < per_class:
            human_texts.extend(human_texts[:per_class - len(human_texts)])

    # fallback for AI: attempt to synthesize
    if len(ai_texts) < per_class:
        print(f"Only {len(ai_texts)} real AI samples found. Synthesizing {per_class - len(ai_texts)} AI-like samples as fallback.")
        synthesized = synthetic_ai_from_human(human_texts, per_class - len(ai_texts))
        ai_texts.extend(synthesized)

    human_texts = human_texts[:per_class]
    ai_texts = ai_texts[:per_class]

    # create DataFrame and clean/deduplicate
    df = pd.DataFrame({"text": human_texts + ai_texts, "label": [0] * per_class + [1] * per_class})
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 30]  # safety filter
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # If de-dup reduced below target, pad by sampling human texts (rare)
    while len(df) < total_size:
        pad = df.sample(1, random_state=len(df)).copy()
        df = pd.concat([df, pad], ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} (shape {df.shape})")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5000, help="Total dataset size (even number)")
    parser.add_argument("--out", type=str, default="dataset.csv", help="Output CSV filename")
    args = parser.parse_args()
    if args.size % 2 != 0:
        print("Please pass an even size (so classes are balanced).")
        sys.exit(1)
    df = build_dataset(total_size=args.size, out_path=args.out)
    print(df.head())
