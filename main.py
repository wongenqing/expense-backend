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

# model config
MODEL_PATH = "model"
MODEL_ZIP = "model.zip"
MODEL_URL = "https://drive.google.com/uc?id=1Bv76nF8tQtvfTPKl6L_J2eat_zCGQDNg"

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# malaysia timezone
TIMEZONE = pytz.timezone("Asia/Kuala_Lumpur")

# global variables (loaded once)
model = None
tokenizer = None
label_map = None


def load_model():
    global model, tokenizer, label_map

    # avoid loading multiple times
    if model is not None:
        return

    print("Loading model...")

    # download model if not exist
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")

    print("Model folder:", os.listdir(MODEL_PATH))

    # load trained model
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # load label map
    with open(f"{MODEL_PATH}/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)

    print("Model ready")


# load spaCy for entity detection
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# create API
app = FastAPI(title="Expense NLP API")


# request format
class TextInput(BaseModel):
    text: str


# load model when server starts
@app.on_event("startup")
def startup_event():
    load_model()


# simple health check
@app.get("/")
def root():
    return {"message": "API is running"}


# get current malaysia time
def get_now():
    return datetime.now(TIMEZONE)


# format datetime nicely
def format_datetime(dt):
    return dt.strftime("%B %d, %Y at %I:%M:%S %p UTC+8")


# remove extra spaces only
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


# extract amount
def extract_amount(text):
    match = re.search(r"(?:rm\s*)?(\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)
    return float(match.group(1)) if match else None

# extact date
def extract_date(text):
    raw = text.strip()

    # fix common speech-to-text mistakes
    fixed = raw.lower()
    fixed = fixed.replace("yester day", "yesterday")
    fixed = fixed.replace("to day", "today")
    fixed = fixed.replace("tomor row", "tomorrow")

    now = get_now()

    # if user clearly says yesterday
    if "yesterday" in fixed:
        return format_datetime(now - timedelta(days=1))

    # if user says today
    if "today" in fixed:
        return format_datetime(now)

    # if user says tomorrow
    if "tomorrow" in fixed:
        return format_datetime(now + timedelta(days=1))

    # try to detect actual date from text
    results = search_dates(
        raw,
        settings={'PREFER_DATES_FROM': 'past'}
    )

    if results:
        detected_text, date_obj = results[0]

        # only accept if it looks like a real date (not random word)
        if re.search(r'\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', detected_text.lower()):

            # convert to malaysia timezone
            if date_obj.tzinfo is None:
                date_obj = TIMEZONE.localize(date_obj)
            else:
                date_obj = date_obj.astimezone(TIMEZONE)

            return format_datetime(date_obj)

    # no date found → just use today
    return format_datetime(now)


# extract merchant
def extract_merchant(text):

    # try regex first
    match = re.search(
        r"(?:at|from|in|to|paid|pay|for)\s+([A-Za-z][A-Za-z0-9&'\/\-\s]*)",
        text,
        re.IGNORECASE
    )

    if match:
        merchant = match.group(1)

        # remove trailing noise
        merchant = re.sub(
            r"\b(for|on|with|using|and|yesterday|today|tomorrow).*",
            "",
            merchant,
            flags=re.IGNORECASE
        )

        return merchant.strip()

    # fallback to spaCy
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "FAC", "PRODUCT"]:
            return ent.text

    return None


# predict category using model
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

    return {
        "label": label_map[str(pred_id.item())],
        "confidence": round(confidence.item(), 4)
    }


# main API
@app.post("/predict")
def predict(input_data: TextInput):

    text = input_data.text.strip()

    category_result = predict_category(text)

    # fallback if confidence too low
    final_category = category_result["label"]
    if category_result["confidence"] < 0.60:
        final_category = "Others"

    return {
        "text": text,
        "amount": extract_amount(text),
        "date": extract_date(text),
        "merchant": extract_merchant(text),
        "category": final_category,
        "confidence": category_result["confidence"]
    }