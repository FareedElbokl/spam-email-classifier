# Spam Email Classifier

A machine learning project in Python that classifies SMS messages as spam or non-spam by training and using a **Multinomial Naive Bayes** model with **TF-IDF** features.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Overview

In this project, I built a spam email (SMS) classifier using Python and machine learning. Started by loading and preprocessing the dataset, cleaning the text by lowercasing and removing punctuation. Then, converted the messages into numerical features using TF-IDF vectorization, which captures how important each word is relative to all the messages. Then trained a Multinomial Naive Bayes model on these features to distinguish between spam and non-spam (ham) messages. After splitting the dataset into training and test sets, I evaluated the model’s performance using metrics like accuracy, precision, recall, and F1 score. Finally, saved both the trained model and the TF-IDF vectorizer so that new messages can be transformed and classified without retraining, completing a full machine learning workflow from raw data to predictions.

1. **Load and preprocess data** – clean and normalize SMS messages.
2. **Vectorize text** – convert messages into numerical features using TF-IDF.
3. **Train model** – use a Naive Bayes classifier to distinguish spam from ham (non-spam).
4. **Evaluate** – compute accuracy, precision, recall, and F1 score.
5. **Predict** – classify new SMS messages.

---

## Dataset

- Source: [UCI Machine Learning Repository – SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Format: Tab-separated file with columns `label` and `message`
- Labels:
  - `ham` = non-spam message
  - `spam` = spam message

---

## Features

- **Text preprocessing:** lowercase, remove punctuation, clean text.
- **TF-IDF vectorization:** converts messages into numerical features for the classifier.
- **Naive Bayes classifier:** MultinomialNB for text classification.
- **Model persistence:** trained model and TF-IDF vectorizer saved as `.pkl` files for later use.

---

## Requirements

- Python 3.12+
- Libraries: pandas, scikit-learn, nltk, joblib

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd spam-email-classifier
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure the dataset file SMSSpamCollection is in the data/ folder.

---

## Usage

Run the main script to train the model and evaluate it:

```bash
python3 src/main.py
```
