import pandas as pd
import random
from transformers import pipeline

# ===========================================
# CONFIG
# ===========================================
TOTAL_SAMPLES = 5000      # final dataset size
AI_GEN_MODEL = "gpt2"     # local text generator
MIN_LEN = 60
random.seed(42)

# ===========================================
# Load GPT-2 for AI text generation
# ===========================================
print("Loading GPT-2 generator...")
generator = pipeline("text-generation", model=AI_GEN_MODEL, tokenizer=AI_GEN_MODEL)

def generate_ai_text(prompt):
    """Generate AI text using GPT-2."""
    try:
        out = generator(prompt, max_length=120, num_return_sequences=1)
        return out[0]["generated_text"]
    except:
        return ""

# ===========================================
# Human text corpus (safe & offline)
# ===========================================
human_corpus = [
    "The festival brought together people from different cultures, sharing food, music, and laughter.",
    "Time management is one of the biggest challenges students face during their academic journey.",
    "The Indian startup ecosystem has grown rapidly, attracting investments from global companies.",
    "Reading regularly enhances vocabulary and broadens perspective across various subjects.",
    "UPI payments revolutionized the digital economy by enabling fast, secure, and cashless transactions.",
    "Sports foster teamwork, discipline, and a healthy competitive spirit among individuals.",
    "Climate change has become a critical global issue requiring immediate action.",
    "Traveling introduces you to new experiences, cultures, and forms of self-discovery.",
]

# ===========================================
# Build dataset
# ===========================================
data = {"text": [], "label": []}

print("Generating dataset...")

for i in range(TOTAL_SAMPLES // 2):
    # AI text
    topic = random.choice(["technology", "nature", "psychology", "finance", "college life"])
    ai_text = generate_ai_text(f"Write a detailed paragraph about {topic}.")
    
    if len(ai_text) >= MIN_LEN:
        data["text"].append(ai_text)
        data["label"].append(1)

    # Human text
    ht = random.choice(human_corpus)
    data["text"].append(ht)
    data["label"].append(0)

df = pd.DataFrame(data)
df = df.drop_duplicates("text")
df = df[df["text"].str.len() > MIN_LEN]
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv("dataset.csv", index=False)
print("Saved dataset.csv! Shape:", df.shape)
