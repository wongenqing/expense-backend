# =========================================
# IMPORTS
# =========================================
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
MODEL_PATH = "model"
MODEL_ZIP = "model.zip"

MODEL_URL = "https://drive.google.com/uc?id=1Bv76nF8tQtvfTPKl6L_J2eat_zCGQDNg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMEZONE = pytz.timezone("Asia/Kuala_Lumpur")

# =========================================
# GLOBAL MODEL (LAZY LOAD)
# =========================================
model = None
tokenizer = None
label_map = None


def load_model():
    global model, tokenizer, label_map

    if model is not None:
        return

    print("⬇️ Loading model on demand...")

    # ✅ Download model only if not exists
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        print("📦 Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)

        print("✅ Model extracted!")

    # ✅ Load model
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    with open(f"{MODEL_PATH}/label_map.json") as f:
        label_map = json.load(f)

    print("✅ Model loaded successfully!")


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
# FASTAPI INIT
# =========================================
app = FastAPI(title="Expense NLP API 🇲🇾🚀")


class TextInput(BaseModel):
    text: str


# =========================================
# UTIL FUNCTIONS
# =========================================
def get_today():
    return datetime.now(TIMEZONE)


def extract_amount(text):
    match = re.search(r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)', text.lower())
    return float(match.group(1)) if match else None


def format_datetime(dt):
    return dt.strftime("%B %d, %Y at %I:%M:%S %p UTC+8")


def extract_date(text):
    text_lower = re.sub(r'[^\w\s]', '', text.lower().strip())

    text_lower = text_lower.replace("yester day", "yesterday")
    text_lower = text_lower.replace("to day", "today")

    today = get_today()

    if "yesterday" in text_lower:
        return format_datetime(today - timedelta(days=1))
    if "today" in text_lower:
        return format_datetime(today)
    if "tomorrow" in text_lower:
        return format_datetime(today + timedelta(days=1))

    clean_text = re.sub(r'rm\s*\d+(\.\d{1,2})?', '', text_lower)
    clean_text = re.sub(r'\b\d+\b', '', clean_text)

    results = search_dates(clean_text, settings={'PREFER_DATES_FROM': 'past'})

    if results:
        for _, date_obj in results:
            date_obj = TIMEZONE.localize(date_obj) if date_obj.tzinfo is None else date_obj.astimezone(TIMEZONE)
            return format_datetime(date_obj)

    return format_datetime(today)


def extract_merchant(text):
    match = re.search(r'(?:at|from|in)\s+([A-Za-z][A-Za-z0-9&\'\-\s]*)', text, re.IGNORECASE)
    if match:
        merchant = re.sub(r'\b(for|on|with|and|using|yesterday|today|tomorrow).*', '', match.group(1), re.IGNORECASE)
        return merchant.strip()

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "FAC"]:
            return ent.text

    return None


def predict_category(text):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = outputs.logits.argmax().item()
    return label_map[str(pred_id)]


def process_expense(text):
    return {
        "original_text": text,
        "amount": extract_amount(text),
        "date": extract_date(text),
        "merchant": extract_merchant(text),
        "category": predict_category(text)
    }


# =========================================
# API
# =========================================
@app.post("/predict")
def predict(input: TextInput):
    return process_expense(input.text)


@app.get("/")
def root():
    return {"message": "Expense NLP API is running 🇲🇾🚀"}