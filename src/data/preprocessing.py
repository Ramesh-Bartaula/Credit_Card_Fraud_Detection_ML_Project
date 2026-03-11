import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
from pathlib import Path

## Getting project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_yaml(file_name):
    config_path = PROJECT_ROOT / 'configs' / file_name
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


## Loading both files
paths_config = load_yaml('config.yaml')
params_config = load_yaml('model_params.yaml')

class SimpleFraudPreprocessor:

    def __init__(self):
        self.scaler = RobustScaler()
        ## Separate encoders for location and amount so they don't overwrite each other
        self.type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()

    def prepare_features(self, df):
        ## Converting raw dataframe to features

        data = df.copy()

        ## Converting TransactionDate to datetime python data object
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

        ## Creating time features
        data['Hour'] = data['TransactionDate'].dt.hour
        data['DayOfWeek'] = data['TransactionDate'].dt.dayofweek
        data['Month'] = data['TransactionDate'].dt.month

        ## Using the specific encoder for TransactionType for purchase and refund 0s and qs
        data['TransactionType_encoded'] = self.type_encoder.fit_transform(data['TransactionType'])

        ## Use the specific encoder for Location
        data['Location_encoded'] = self.location_encoder.fit_transform(data['Location'])

        ## Selecting features for modeling
        feature_columns = ['Amount', 'Hour', 'DayOfWeek', 'Month',
                           'TransactionType_encoded', 'Location_encoded']

        X = data[feature_columns]
        y = data['IsFraud']

        print(f"Created {len(feature_columns)} features")
        return X, y

    def split_data(self, X, y, test_size=0.3):
        ## splitting data into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        ## Scaling 'Amount' to handle outliers

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        ## Fit on train, transform both (prevents data leakage)
        X_train_scaled[['Amount']] = self.scaler.fit_transform(X_train[['Amount']])
        X_test_scaled[['Amount']] = self.scaler.transform(X_test[['Amount']])

        return X_train_scaled, X_test_scaled

    def save_preprocessor(self):
        ## Saving all encoders and scaler in one file

        save_path = PROJECT_ROOT / paths_config['paths']['processed_artifacts']

        save_path.parent.mkdir(parents=True, exist_ok=True)

        ## Creating a dictionary to hold everything
        artifacts = {
            'scaler': self.scaler,
            'type_encoder': self.type_encoder,
            'location_encoder': self.location_encoder
        }
        joblib.dump(artifacts, save_path)
        print(f"Preprocessor saved to {save_path}")

if __name__ == "__main__":

    # Pulling raw data path from config.yaml
    DATA_PATH = PROJECT_ROOT / paths_config['paths']['raw_data']

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    ## Preprocessing steps
    prep = SimpleFraudPreprocessor()
    X, y = prep.prepare_features(df)
    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    ## Saving
    prep.save_preprocessor()
    print("Done! Data ready for modeling.")