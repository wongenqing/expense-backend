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

    # ✅ Download model at runtime
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_ZIP, quiet=False)

        print("📦 Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)

    # ✅ Load model
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


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "API is running 🚀"}


# =========================================
# FUNCTIONS
# =========================================
def extract_amount(text):
    match = re.search(r'(?:rm\s*)?(\d+(?:\.\d{1,2})?)', text.lower())
    return float(match.group(1)) if match else None


def predict_category(text):
    load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = outputs.logits.argmax().item()
    return label_map[str(pred_id)]


@app.post("/predict")
def predict(input: TextInput):
    return {
        "text": input.text,
        "amount": extract_amount(input.text),
        "category": predict_category(input.text)
    }