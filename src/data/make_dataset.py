import pandas as pd
from pathlib import Path
import yaml


## setting up root and loading config
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'



def load_config():
    with open(CONFIG_PATH,'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    ## Getting path from config.yaml
    ## using .parent to get the directory 'data/raw' from the full file path
    raw_data_path = PROJECT_ROOT / config['paths']['raw_data']
    raw_data_dir = raw_data_path.parent

    ## Creating directory if it doesn't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking for dataset at: {raw_data_path}")

    if raw_data_path.exists():
        print("Dataset found!")
        # Optional: Print a small preview to confirm it's readable
        df = pd.read_csv(raw_data_path)
        print(f" Dataset loaded successfully with {len(df)} rows.")
    else:
        print(" Dataset missing!")
        print(f"Please place your 'credit_card_fraud_dataset.csv' file in: {raw_data_dir}")


if __name__ == "__main__":
    main()
