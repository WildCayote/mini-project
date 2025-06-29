from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
import joblib
import pandas as pd

# Add root to sys.path to import your functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.feature_engineering import clean_review, vectorize_reviews

app = FastAPI()

# Setup templates and static folder
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and tfidf once on startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_model.pkl'))
TFIDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl'))

model = joblib.load(MODEL_PATH)
tfidf_path = TFIDF_PATH

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, review: str = Form(...)):
    # Clean input
    cleaned = clean_review(review)
    review_series = pd.Series([cleaned])
    # Vectorize
    X = vectorize_reviews(review_series, path_tfidf=tfidf_path)
    # Predict
    pred_proba = model.predict_proba(X)[0]
    pred_label = model.predict(X)[0]
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence = pred_proba[pred_label]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": sentiment,
        "confidence": f"{confidence:.2f}",
        "input_review": review
    })
