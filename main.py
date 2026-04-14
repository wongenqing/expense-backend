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

# Config
MODEL_PATH = "model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🇲🇾 Malaysia timezone
TIMEZONE = pytz.timezone("Asia/Kuala_Lumpur")


# =========================================
# LOAD MODEL (SAFE FOR RAILWAY)
# =========================================
model = None
tokenizer = None
label_map = None

if os.path.exists(MODEL_PATH):
    print("📂 Model files:", os.listdir(MODEL_PATH))

    print("🔄 Loading model...")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("🔄 Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Load label map
    with open(f"{MODEL_PATH}/label_map.json") as f:
        label_map = json.load(f)

    print("✅ Model loaded successfully!")
else:
    print("⚠️ Model folder not found (Railway will handle later)")


# =========================================
# LOAD SPACY (SAFE FOR RAILWAY)
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


# request format
class TextInput(BaseModel):
    text: str


# =========================================
# UTIL FUNCTIONS
# =========================================
def get_today():
    return datetime.now(TIMEZONE)


# Extract Amount
def extract_amount(text):
    match = re.search(r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)', text.lower())
    if match:
        return float(match.group(1))
    return None


# Extract Date
def format_datetime(dt):
    return dt.strftime("%B %d, %Y at %I:%M:%S %p UTC+8")


def extract_date(text):
    text_lower = text.lower().strip()

    # Remove punctuation
    text_lower = re.sub(r'[^\w\s]', '', text_lower)

    # speech recognition splits
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

    date_keywords = [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec",
        "monday","tuesday","wednesday","thursday",
        "friday","saturday","sunday"
    ]

    if any(k in clean_text for k in date_keywords):
        results = search_dates(
            clean_text,
            settings={'PREFER_DATES_FROM': 'past'}
        )

        if results:
            for _, date_obj in results:
                date_obj = TIMEZONE.localize(date_obj) if date_obj.tzinfo is None else date_obj.astimezone(TIMEZONE)
                return format_datetime(date_obj)

    return format_datetime(today)


# Extract merchant
def extract_merchant(text):
    original_text = text

    match = re.search(
        r'(?:at|from|in)\s+([A-Za-z][A-Za-z0-9&\'\-\s]*)',
        text,
        flags=re.IGNORECASE
    )
    if match:
        merchant = match.group(1).strip()

        merchant = re.sub(
            r'\b(for|on|with|and|using|yesterday|today|tomorrow).*',
            '',
            merchant,
            flags=re.IGNORECASE
        )

        return merchant.strip()

    match = re.search(
        r'(?:on|for)\s+([A-Za-z][A-Za-z0-9&\'\-\s]*)',
        text,
        flags=re.IGNORECASE
    )
    if match:
        merchant = match.group(1).strip()

        if re.match(r'^\d', merchant):
            merchant = re.sub(r'^\d+\s*', '', merchant)

        merchant = re.sub(
            r'\b(at|from|in|with|and|yesterday|today|tomorrow).*',
            '',
            merchant,
            flags=re.IGNORECASE
        )

        return merchant.strip()

    # spaCy fallback
    doc = nlp(original_text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "FAC"]:
            return ent.text

    return None


# Predict category
def predict_category(text):
    if model is None or tokenizer is None or label_map is None:
        return "Model not available"

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


# Main pipeline
def process_expense(text):
    return {
        "original_text": text,
        "amount": extract_amount(text),
        "date": extract_date(text),
        "merchant": extract_merchant(text), 
        "category": predict_category(text)  
    }


# =========================================
# API ENDPOINTS
# =========================================
@app.post("/predict")
def predict(input: TextInput):
    return process_expense(input.text)


@app.get("/")
def root():
    return {"message": "Expense NLP API is running 🇲🇾🚀"}