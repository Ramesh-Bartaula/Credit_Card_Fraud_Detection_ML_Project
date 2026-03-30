
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


## Loading both config files
paths_config = load_yaml('config.yaml')
params_config = load_yaml('model_params.yaml')


class SimpleFraudPreprocessor:

    def __init__(self):
        self.scaler = RobustScaler()
        self.type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self._encoders_fitted = False

        ## Merchant stats — computed on training data, applied to test/inference
        ## Storing these prevents data leakage from test set into merchant features
        self.merchant_fraud_rate = {}
        self.merchant_tx_count = {}
        self.merchant_avg_amount = {}
        self.global_fraud_rate = 0.0
        self.global_avg_amount = 0.0

    def _compute_merchant_stats(self, df):
        """
        Compute per-merchant statistics from training data only.
        These are stored and reused at test/inference time to prevent leakage.
        """
        self.merchant_fraud_rate  = df.groupby('MerchantID')['IsFraud'].mean().to_dict()
        self.merchant_tx_count    = df.groupby('MerchantID')['Amount'].count().to_dict()
        self.merchant_avg_amount  = df.groupby('MerchantID')['Amount'].mean().to_dict()

        ## Global fallbacks for unseen merchants at inference time
        self.global_fraud_rate  = df['IsFraud'].mean()
        self.global_avg_amount  = df['Amount'].mean()

        print(f"Computed stats for {len(self.merchant_fraud_rate)} unique merchants")

    def _apply_merchant_features(self, data):
        """
        Apply stored merchant statistics to a dataframe.
        Uses global fallback for any merchant not seen during training.
        """
        data['Merchant_fraud_rate'] = (
            data['MerchantID']
            .map(self.merchant_fraud_rate)
            .fillna(self.global_fraud_rate)
        )
        data['Merchant_tx_count'] = (
            data['MerchantID']
            .map(self.merchant_tx_count)
            .fillna(1)
        )
        data['Merchant_avg_amount'] = (
            data['MerchantID']
            .map(self.merchant_avg_amount)
            .fillna(self.global_avg_amount)
        )

        ## How much does this transaction deviate from the merchant's normal amount?
        ## A $4000 transaction at a merchant that usually sees $20 is suspicious
        data['Amount_vs_merchant_avg'] = data['Amount'] - data['Merchant_avg_amount']

        return data

    def prepare_features(self, df, fit_encoders=True):
        """Convert raw dataframe to model-ready features.

        Args:
            df: Raw input dataframe
            fit_encoders: True during training, False during inference to prevent data leakage
        """
        data = df.copy()

        ## Converting TransactionDate to datetime
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

        ## -------------------------------------------------------
        ## TIME FEATURES
        ## -------------------------------------------------------
        data['Hour']      = data['TransactionDate'].dt.hour
        data['DayOfWeek'] = data['TransactionDate'].dt.dayofweek
        data['Month']     = data['TransactionDate'].dt.month

        ## Is the transaction happening late at night? (higher fraud risk)
        data['Is_night']   = data['Hour'].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)

        ## Is it a weekend?
        data['Is_weekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        ## -------------------------------------------------------
        ## MERCHANT FEATURES — most valuable new signal
        ## -------------------------------------------------------
        if fit_encoders:
            ## Training: compute stats from this data, then apply
            self._compute_merchant_stats(data)
        else:
            ## Inference: use pre-computed stats from training
            if not self._encoders_fitted:
                raise RuntimeError(
                    "Preprocessor has not been fitted. "
                    "Call prepare_features(fit_encoders=True) on training data first."
                )

        data = self._apply_merchant_features(data)

        ## -------------------------------------------------------
        ## CATEGORICAL ENCODING
        ## -------------------------------------------------------
        if fit_encoders:
            data['TransactionType_encoded'] = self.type_encoder.fit_transform(data['TransactionType'])
            data['Location_encoded']        = self.location_encoder.fit_transform(data['Location'])
            self._encoders_fitted = True
        else:
            data['TransactionType_encoded'] = self.type_encoder.transform(data['TransactionType'])
            data['Location_encoded']        = self.location_encoder.transform(data['Location'])

        ## -------------------------------------------------------
        ## FEATURE SELECTION
        ## Original 6 features + 6 new merchant/time features = 12 total
        ## -------------------------------------------------------
        feature_columns = [
            ## Original features
            'Amount',
            'Hour',
            'DayOfWeek',
            'Month',
            'TransactionType_encoded',
            'Location_encoded',

            ## New time features
            'Is_night',
            'Is_weekend',

            ## New merchant features (the key improvement)
            'Merchant_fraud_rate',       ## How risky is this merchant historically?
            'Merchant_tx_count',         ## How many transactions does this merchant have?
            'Merchant_avg_amount',       ## What is the normal spend at this merchant?
            'Amount_vs_merchant_avg',    ## Is this transaction unusually large?
        ]

        X = data[feature_columns]
        y = data['IsFraud']

        print(f"Created {len(feature_columns)} features")
        return X, y

    def split_data(self, X, y, test_size=0.3):
        """Split into train/test with stratification to preserve fraud ratio."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
        print(f"Fraud in train: {y_train.sum()} | Fraud in test: {y_test.sum()}")
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Scale continuous amount features using RobustScaler.

        RobustScaler uses median and IQR — not affected by extreme outliers
        which are common in fraud transaction amounts.
        Fits only on training data to prevent data leakage.
        """
        X_train_scaled = X_train.copy()
        X_test_scaled  = X_test.copy()

        ## Scale all continuous amount-related columns
        cols_to_scale = ['Amount', 'Merchant_avg_amount', 'Amount_vs_merchant_avg']

        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale]  = self.scaler.transform(X_test[cols_to_scale])

        return X_train_scaled, X_test_scaled

    def save_preprocessor(self):
        """Save all encoders, scaler, and merchant stats to a single artifact file."""
        save_path = PROJECT_ROOT / paths_config['paths']['processed_artifacts']
        save_path.parent.mkdir(parents=True, exist_ok=True)

        artifacts = {
            'scaler':               self.scaler,
            'type_encoder':         self.type_encoder,
            'location_encoder':     self.location_encoder,
            'merchant_fraud_rate':  self.merchant_fraud_rate,
            'merchant_tx_count':    self.merchant_tx_count,
            'merchant_avg_amount':  self.merchant_avg_amount,
            'global_fraud_rate':    self.global_fraud_rate,
            'global_avg_amount':    self.global_avg_amount,
        }
        joblib.dump(artifacts, save_path)
        print(f"Preprocessor saved to {save_path}")

    def load_preprocessor(self, load_path=None):
        """Load saved preprocessor artifacts for inference."""
        if load_path is None:
            load_path = PROJECT_ROOT / paths_config['paths']['processed_artifacts']

        artifacts = joblib.load(load_path)

        self.scaler              = artifacts['scaler']
        self.type_encoder        = artifacts['type_encoder']
        self.location_encoder    = artifacts['location_encoder']
        self.merchant_fraud_rate = artifacts['merchant_fraud_rate']
        self.merchant_tx_count   = artifacts['merchant_tx_count']
        self.merchant_avg_amount = artifacts['merchant_avg_amount']
        self.global_fraud_rate   = artifacts['global_fraud_rate']
        self.global_avg_amount   = artifacts['global_avg_amount']
        self._encoders_fitted    = True

        print(f"Preprocessor loaded from {load_path}")


if __name__ == "__main__":

    DATA_PATH = PROJECT_ROOT / paths_config['paths']['raw_data']

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    prep = SimpleFraudPreprocessor()
    X, y = prep.prepare_features(df, fit_encoders=True)
    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    prep.save_preprocessor()
    print("Done! Data ready for modeling.")