import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


def main():
    # Create directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Checking for dataset at {RAW_DATA_DIR}...")

    if (RAW_DATA_DIR / 'credit_card_fraud_dataset.csv').exists():
        print("Dataset found!")
    else:
        print("Dataset missing. Please place credit_card_fraud_dataset.csv in data/raw/")


if __name__ == "_main_":
    main()
