# Voxpense - Expense Backend API

This is the backend service for the Voxpense expense tracking application. It processes user input and extracts structured expense data using Natural Language Processing (NLP).

## Features

* Extract amount, date, merchant, and category from text
* NLP-based category classification using RoBERTa
* Named Entity Recognition using spaCy
* Date parsing with support for relative dates (e.g., "yesterday")
* REST API built with FastAPI

## Tech Stack

* FastAPI
* Python
* Transformers (RoBERTa)
* spaCy
* PyTorch
* dateparser

## API Endpoint

POST /predict

### Example Request

```json
{
  "text": "I spent RM20 on lunch yesterday"
}
```

### Example Response

```json
{
  "text": "I spent RM20 on lunch yesterday",
  "amount": 20,
  "date": "2026-04-15",
  "merchant": null,
  "category": "Food & Drink",
  "confidence": 0.99
}
```

## Deployment

Hosted on Railway:
https://web-production-0dbaca.up.railway.app

## Author

Wong En Qing
