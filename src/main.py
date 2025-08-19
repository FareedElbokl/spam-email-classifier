# src/main.py

from preprocessing import load_data, preprocess_dataframe
from vectorizer import fit_vectorizer, transform_vectorizer
from train import train_model
from evaluate import evaluate_model

from sklearn.model_selection import train_test_split

def main():
    # 1. Load and preprocess the data
    df = load_data()
    df = preprocess_dataframe(df)

    # 2. Convert labels to numeric: ham=0, spam=1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # 3. Fit TF-IDF vectorizer on cleaned messages
    X, vectorizer = fit_vectorizer(df['clean_message'])

    # 4. Prepare labels
    y = df['label_num']

    # 5. Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train the model
    model = train_model(X_train, y_train)

    # 7. Predict on the test set
    y_pred = model.predict(X_test)

    # 8. Evaluate model performance
    metrics = evaluate_model(y_test, y_pred)
    print("Model evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 9. Example: Transform new messages and predict
    new_messages = [
        "Congratulations! You've won a $1000 gift card. Claim now!",
        "Hey, are we still meeting for lunch today?"
    ]
    X_new = transform_vectorizer(new_messages)
    predictions = model.predict(X_new)
    print("\nPredictions for new messages:")
    for msg, pred in zip(new_messages, predictions):
        label = "spam" if pred == 1 else "ham"
        print(f"{msg} -> {label}")


if __name__ == "__main__":
    main()
