import joblib
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_review(text: str) -> str:
    """
    Clean the review text by performing the following steps:
    1. Lowercasing the text
    2. Removing punctuation
    3. Removing stopwords
    4. Removing 'br' tags
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = re.findall(r'\b\w+\b', text)
    # Remove 'br' and stopwords
    cleaned = [word for word in words if word not in stop_words and word != 'br']
    return ' '.join(cleaned)

def vectorize_reviews(reviews: pd.Series, path_tfidf: str) -> pd.DataFrame:
    """
    Vectorize the cleaned reviews using TF-IDF.
    """
    tfidf = joblib.load(open(path_tfidf, 'rb'))
    X = tfidf.transform(reviews)
    return X

def convert_sentiment_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert sentiment labels to binary labels.
    """
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df