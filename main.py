
from src.data.preprocessing import SimpleFraudPreprocessor
from src.models.train_model import train_fraud_model
import pandas as pd
import yaml
from pathlib import Path


def main():
    """Main pipeline execution."""

    # Setup paths
    PROJECT_ROOT = Path(__file__).parent

    def load_yaml(file_name):
        config_path = PROJECT_ROOT / 'configs' / file_name
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    try:
        paths_config = load_yaml('config.yaml')
    except FileNotFoundError:
        paths_config = {
            'paths': {
                'raw_data': 'data/raw/fraud_data.csv'
            }
        }

    DATA_PATH = PROJECT_ROOT / paths_config['paths']['raw_data']

    print("\n" + "=" * 60)
    print(" FRAUD DETECTION PIPELINE - STARTING")
    print("=" * 60)


    # PREPROCESSING

    print("\n STEP 1: DATA PREPROCESSING")
    print("-" * 60)

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✓ Dataset loaded: {df.shape}")
        print(f"✓ Fraud cases: {df['IsFraud'].sum()} / {len(df)} ({df['IsFraud'].mean() * 100:.2f}%)")

        prep = SimpleFraudPreprocessor()
        X, y = prep.prepare_features(df, fit_encoders=True)
        X_train, X_test, y_train, y_test = prep.split_data(X, y)
        X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)
        prep.save_preprocessor()

        print("✓ Preprocessing complete!\n")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False


    # MODEL TRAINING

    print("\n STEP 2: MODEL TRAINING & EVALUATION")
    print("-" * 60)

    try:
        train_fraud_model()
        print("\n✓ Model training complete!\n")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False


    # COMPLETION

    print("=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nArtifacts saved:")
    print("  Model: models/fraud_model.joblib")
    print("  Threshold: models/best_threshold.joblib")
    print("  Preprocessor: artifacts/preprocessor.joblib")
    print("  Reports: reports/confusion_matrix.png")
    print("\n" + "=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)