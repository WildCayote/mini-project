# mini-project

A lightweight machine learning pipeline for predicting sentiment (positive or negative) from movie reviews using Logistic Regression and TF-IDF. Built with `scikit-learn`, deployed with `FastAPI`, and tracked using `DVC`.

## Overview

This project processes IMDb-style review data, cleans and vectorizes the text, trains a Logistic Regression classifier, and serves predictions via a FastAPI web UI.

## Tools Used

| Tool         | Purpose                                 |
| ------------ | --------------------------------------- |
| DVC          | For handling data                       |
| Pandas       | For data analysis                       |
| matplotlib   | For visualization                       |
| Seaborn      | For enhancing matplotlib visualizations |
| Numpy        | For numerical operations                |
| Scikit-learn | For machine learning                    |
| FastAPI      | For exposing predictions via an API     |
| Joblib       | For model serialization                 |

## Project Structure

mini-project/
│ images/ # Screenshots of the app
│ ├── home.png # Screenshot of the home page
│ ├── good_review.png # Screenshot of a good review prediction
│ └── bad_review.png # Screenshot of a bad review prediction
|
├── app/ # FastAPI app folder
│ ├── main.py # FastAPI application
│ ├── static/styles.css # Custom CSS for the UI
│ └── templates/index.html # Frontend UI
│
├── data/
│ ├── IMDB Dataset.csv.dvc # DVC-managed data
│
├── models/ # Saved TF-IDF and model
│ ├── logistic_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── scripts/
│ ├── predict.py # Command-line prediction script
│ ├── feature_engineering.py # Cleaning + vectorizing utilities
|
├── notebooks/
│ ├── eda.ipynb # Exploratory Data Analysis
│ └── feature_engineering.ipynb # Feature engineering notebook
│ └── training.ipynb # Model training notebook
├── tests/
│ ├── **init**.py
│ ├── dummy_test.py # Model training pipeline
│
└── README.md
└── requirements.txt # Dependencies

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/mini-project.git
cd mini-project
pip install -r requirements.txt
```

## How to run the fastapi app

```bash
cd app
uvicorn main:app --reload
```

Then open your browser and go to: http://127.0.0.1:8000

Submit a review via the UI and get an instant prediction.

![FastAPI Sentiment App Screenshot](images/home.png)

![Good review screenshot](images/good_review.png)

![Bad review screenshot](images/bad_review.png)

## How to run the command-line prediction script

```bash
python scripts/predict.py "Your movie review here"
```

## Contributor

- **Tinsae Shemalise**
