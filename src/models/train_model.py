
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# Importing preprocessor class
from src.data.preprocessing import SimpleFraudPreprocessor

# Paths setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'credit_card_fraud_dataset.csv'
MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'artifacts' / 'fraud_model.pkl'


def train_fraud_model():
    # 1. Load Data
    print("--- Loading Data ---")
    df = pd.read_csv(DATA_PATH)

    # Preprocess Data using from class
    print("--- Preprocessing ---")
    prep = SimpleFraudPreprocessor()
    X, y = prep.prepare_features(df)
    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    # Initialising Random Forest
    # class_weight='balanced' helps the model focus on the rare fraud cases
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    # Training the model
    print("--- Training Random Forest ---")
    model.fit(X_train_scaled, y_train)

    # Evaluating
    print("--- Evaluation ---")
    y_pred = model.predict(X_test_scaled)

    # This report shows Precision, Recall, and F1-Score
    print(classification_report(y_test, y_pred))

    # Saveing the Model and the Preprocessor artifacts
    print("--- Saving Artifacts ---")
    prep.save_preprocessor()  # Saves scaler and encoders
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_fraud_model()