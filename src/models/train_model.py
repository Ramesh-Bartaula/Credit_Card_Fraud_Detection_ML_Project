
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix , recall_score, accuracy_score
from pathlib import Path

# Importing preprocessor class
from src.data.preprocessing import SimpleFraudPreprocessor

# Paths setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'credit_card_fraud_dataset.csv'
MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'artifacts' / 'fraud_model.pkl'


def train_fraud_model():
    # Loading Data
    print("--- Loading Data ---")
    df = pd.read_csv(DATA_PATH)

    # Preprocess Data using from class
    print("--- Preprocessing ---")
    prep = SimpleFraudPreprocessor()
    X, y = prep.prepare_features(df)
    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)


    #--Baseline Moeling------
    print("\n---Running Baseline Modek------")
    baseline = LogisticRegression()
    baseline.fit(X_train_scaled, y_train)
    baseline_pred = baseline.predict(X_test_scaled)
    baseline_recall = recall_score(y_test , baseline_pred)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

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
    rf_pred = model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)


    ## Comparison between model
    print("\n" + "="*25)
    print("       MODEL COMPARISON      ")
    print("="*25)
    print(f"Metric     | Baseline Model  | Random Forest Model")
    print(f"Accuracy   | {baseline_accuracy:.2f}          | {rf_accuracy:.2f}")
    print(f"Recall     | {rf_recall:.2f}            | {baseline_recall:.2f}")


    ## calculating precision , re-call and f-1 score of rf model
    print("="*25)
    print(f"Classification Report of Random forest model")
    print(classification_report(y_test, rf_pred))
    print(f"Confusion Matrix of Random Forest Model")
    print(confusion_matrix(y_test, rf_pred))

    # Saving the best Model and the Preprocessor artifacts

    prep.save_preprocessor()  # Saves scaler and encoders
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_fraud_model()