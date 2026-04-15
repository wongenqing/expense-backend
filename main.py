# =========================================
# IMPORTS
# =========================================
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaForSequenceClassification, AutoTokenizer
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
import torch.nn.functional as F

# =========================================
# CONFIG
# =========================================
MODEL_PATH = "model"
MODEL_ZIP = "model.zip"
MODEL_URL = "https://drive.google.com/uc?id=1Bv76nF8tQtvfTPKl6L_J2eat_zCGQDNg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMEZONE = pytz.timezone("Asia/Kuala_Lumpur")

# =========================================
# GLOBALS
# =========================================
model = None
tokenizer = None
label_map = None

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    global model, tokenizer, label_map

    if model is not None:
        return

    print("⬇️ Loading model...")

    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        print("📦 Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(".")

    print("📂 Model folder:", os.listdir(MODEL_PATH))

    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # 🔥 FIX: Always use base tokenizer (avoid corruption issues)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

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
app = FastAPI(title="Expense NLP API 🇲🇾🚀")

@app.on_event("startup")
def startup_event():
    load_model()

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

# =========================================
# UTIL FUNCTIONS
# =========================================
def get_today():
    return datetime.now(TIMEZONE)

def format_datetime(dt):
    return dt.strftime("%B %d, %Y at %I:%M:%S %p UTC+8")

# =========================================
# CLEAN TEXT (IMPROVED)
# =========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'rm\s*\d+(\.\d+)?', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# =========================================
# AMOUNT
# =========================================
def extract_amount(text):
    match = re.search(r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)', text.lower())
    return float(match.group(1)) if match else None

# =========================================
# DATE (🔥 FIXED LOGIC)
# =========================================
def extract_date(text):
    text_lower = text.lower().strip()
    text_lower = re.sub(r'[^\w\s]', '', text_lower)

    text_lower = text_lower.replace("yester day", "yesterday")
    text_lower = text_lower.replace("to day", "today")

    today = get_today()

    # explicit
    if "yesterday" in text_lower:
        return format_datetime(today - timedelta(days=1))
    if "today" in text_lower:
        return format_datetime(today)
    if "tomorrow" in text_lower:
        return format_datetime(today + timedelta(days=1))

    # remove noise
    clean_text_ = re.sub(r'rm\s*\d+(\.\d{1,2})?', '', text_lower)
    clean_text_ = re.sub(r'\b\d+\b', '', clean_text_)

    date_keywords = [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec",
        "monday","tuesday","wednesday","thursday",
        "friday","saturday","sunday"
    ]

    if any(k in clean_text_ for k in date_keywords):
        results = search_dates(clean_text_, settings={'PREFER_DATES_FROM': 'past'})
        if results:
            _, date_obj = results[0]
            date_obj = TIMEZONE.localize(date_obj) if date_obj.tzinfo is None else date_obj.astimezone(TIMEZONE)
            return format_datetime(date_obj)

    return format_datetime(today)

# =========================================
# MERCHANT (🔥 IMPROVED)
# =========================================
KNOWN_MERCHANTS = [
    "kfc","mcdonald","mcd","starbucks","tealive",
    "petronas","shell","grab","shopee","lazada"
]

def extract_merchant(text):
    text_lower = text.lower()

    # ✅ known merchants (but return ORIGINAL text version)
    for m in KNOWN_MERCHANTS:
        if m in text_lower:
            # find exact match in original text
            pattern = re.compile(re.escape(m), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                return match.group(0)  # 🔥 original casing

    # ✅ regex extraction (already uses original text)
    match = re.search(
        r'(?:at|from|in)\s+([A-Za-z][A-Za-z0-9&\'\-\s]*)',
        text,
        re.IGNORECASE
    )
    if match:
        merchant = re.sub(
            r'\b(for|on|with|and|using|yesterday|today|tomorrow).*',
            '',
            match.group(1),
            flags=re.IGNORECASE
        )
        return merchant.strip()  # 🔥 keep original casing

    # ✅ spaCy fallback (already original)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "FAC"]:
            return ent.text  # 🔥 original casing

    return None

# =========================================
# RULE-BASED CATEGORY
# =========================================
def rule_based_category(text):
    text = text.lower()

    rules = {
        "Food & Drink": ["food","lunch","dinner","kfc","mcd","coffee","starbucks"],
        "Transportation": ["grab","taxi","petrol","fuel","toll","parking"],
        "Shopping": ["shopping","lazada","shopee","mall"],
        "Groceries": ["grocery","tesco","aeon","giant"],
        "Utilities": ["electric","water","wifi","internet"],
        "Entertainment": ["movie","netflix","spotify"],
        "Healthcare": ["clinic","hospital","pharmacy"],
        "Education": ["school","course","book"],
        "Rent": ["rent"]
    }

    for category, keywords in rules.items():
        if any(word in text for word in keywords):
            return category

    return None

# =========================================
# PREDICT CATEGORY (🔥 IMPROVED)
# =========================================
def predict_category(text):
    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, pred_id = torch.max(probs, dim=1)

    model_category = label_map[str(pred_id.item())]

    rule_category = rule_based_category(text)

    # ✅ smarter fallback
    if confidence.item() < 0.65 and rule_category:
        return rule_category

    return model_category

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