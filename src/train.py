# src/train.py

import os
import joblib
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_classifier.pkl")

def train_model(X_train, y_train, save_path=DEFAULT_MODEL_PATH):
    """
    Train a Multinomial Naive Bayes model on the training data and save it.

    Args:
        X_train: TF-IDF feature matrix for training
        y_train: labels for training data
        save_path: path to save the trained model

    Returns:
        model: trained MultinomialNB model
    """
    # Initialize and train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the trained model
    joblib.dump(model, save_path)
    print(f"Trained model saved to {save_path}")

    return model
