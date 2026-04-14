from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch
import json
import re
from dateparser.search import search_dates
from datetime import datetime, timedelta
import pytz
import spacy
import os
import subprocess
import gdown
import zipfile

# =========================================
# CONFIG
# =========================================
MODEL_PATH = "model"   # ✅ FIXED
MODEL_ZIP = "model.zip"

MODEL_URL = "https://drive.google.com/uc?id=1Bv76nF8tQtvfTPKl6L_J2eat_zCGQDNg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMEZONE = pytz.timezone("Asia/Kuala_Lumpur")

# =========================================
# GLOBAL MODEL
# =========================================
model = None
tokenizer = None
label_map = None


def load_model():
    global model, tokenizer, label_map

    if model is not None:
        return

    print("⬇️ Loading model...")

    # Download if not exist
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        print("📦 Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")   # ✅ FIXED

    # 🔍 DEBUG (optional)
    print("📂 Model folder:", os.listdir(MODEL_PATH))

    # Load model
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    with open(f"{MODEL_PATH}/label_map.json") as f:
        label_map = json.load(f)

    print("✅ Model ready!")


# =========================================
# LOAD SPACY
# =========================================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("⬇️ Downloading spaCy model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# =========================================
# FASTAPI
# =========================================
app = FastAPI()


@app.on_event("startup")
def startup_event():
    load_model()


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API is running 🚀"}


# =========================================
# FUNCTIONS
# =========================================
def get_today():
    return datetime.now(TIMEZONE)


def extract_amount(text):
    match = re.search(r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)', text.lower())
    return float(match.group(1)) if match else None


def format_datetime(dt):
    return dt.strftime("%B %d, %Y at %I:%M:%S %p UTC+8")


def extract_date(text):
    text_lower = re.sub(r'[^\w\s]', '', text.lower())
    today = get_today()

    if "yesterday" in text_lower:
        return format_datetime(today - timedelta(days=1))
    if "today" in text_lower:
        return format_datetime(today)
    if "tomorrow" in text_lower:
        return format_datetime(today + timedelta(days=1))

    results = search_dates(text_lower)
    if results:
        return format_datetime(results[0][1])

    return format_datetime(today)


def extract_merchant(text):
    # Capture after "at/from/in"
    match = re.search(r'(?:at|from|in)\s+([A-Za-z][A-Za-z0-9&\'\-\s]*)', text, re.IGNORECASE)
    
    if match:
        merchant = match.group(1).strip()

        # ❗ Remove trailing keywords
        merchant = re.sub(
            r'\b(yesterday|today|tomorrow|for|on|with|and|using)\b.*',
            '',
            merchant,
            flags=re.IGNORECASE
        )

        return merchant.strip()

    # fallback spaCy
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:
            return ent.text

    return None


def predict_category(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = outputs.logits.argmax().item()
    return label_map[str(pred_id)]


# =========================================
# API
# =========================================
@app.post("/predict")
def predict(input: TextInput):
    return {
        "text": input.text,
        "amount": extract_amount(input.text),
        "date": extract_date(input.text),
        "merchant": extract_merchant(input.text),
        "category": predict_category(input.text)
    }