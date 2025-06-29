import sys
import os
import joblib
import pandas as pd

# Append project root to sys.path so we can import clean_review, vectorize_reviews
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_engineering import clean_review, vectorize_reviews

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        sys.exit(1)

    input_text = sys.argv[1]

    # Clean the input
    cleaned_text = clean_review(input_text)

    # Create a dummy dataframe to reuse vectorize_reviews (expects a pd.Series)
    review_series = pd.Series([cleaned_text])

    # Load TF-IDF vectorizer and transform input
    tfidf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl'))
    X = vectorize_reviews(review_series, path_tfidf=tfidf_path)

    # Load the trained logistic model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_model.pkl'))
    model = joblib.load(model_path)

    # Make prediction
    pred_proba = model.predict_proba(X)[0]
    pred_label = model.predict(X)[0]

    sentiment = "positive" if pred_label == 1 else "negative"
    confidence = pred_proba[pred_label]

    print(f"Prediction: {sentiment} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
