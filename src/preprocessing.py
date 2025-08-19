# src/preprocessing.py

import pandas as pd
import os
import string

def load_data(filename="SMSSpamCollection"):
    """
    Load the SMS Spam dataset into a Pandas DataFrame.
    Works regardless of current working directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", filename)
    df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    return df

def clean_text(text):
    """
    Clean a single SMS message:
    - Lowercase
    - Remove punctuation
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text

def preprocess_dataframe(df):
    """
    Apply text cleaning to all messages in the DataFrame.
    """
    df['clean_message'] = df['message'].apply(clean_text)
    return df
