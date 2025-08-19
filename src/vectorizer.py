# src/vectorizer.py

# fit_vectorizer -> "learn words, create feature space, encode training messages"
# transform_vectorizer -> "encode new messages in the same feature space"

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VECT_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

def fit_vectorizer(messages, save_path=DEFAULT_VECT_PATH):
    """
    Fit a TF-IDF vectorizer on the training messages and save it.

    Args:
        messages (list or pd.Series): list of cleaned messages
        save_path (str): path to save the fitted vectorizer
    Returns:
        X (sparse matrix): TF-IDF vectors
        vectorizer (TfidfVectorizer): the fitted vectorizer
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(messages)

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the vectorizer
    joblib.dump(vectorizer, save_path)
    return X, vectorizer

def transform_vectorizer(messages, vectorizer_path=DEFAULT_VECT_PATH):
    """
    Transform new messages into TF-IDF vectors using a saved vectorizer.

    Args:
        messages (list or pd.Series): list of cleaned messages
        vectorizer_path (str): path to the saved vectorizer
    Returns:
        X (sparse matrix): TF-IDF vectors
    """
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(messages)
    return X
