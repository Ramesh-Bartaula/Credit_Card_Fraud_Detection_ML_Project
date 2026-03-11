
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import joblib
from datetime import datetime
from pathlib import Path


## Getting  project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'credit_card_fraud_dataset.csv'


class SimpleFraudPreprocessor:

    def __init__(self):
        self.scaler = None
        self.label_encoder = LabelEncoder()

    def prepare_features(self, df):
        """Converting raw dataframe to features"""
        ## Making a copy
        data = df.copy()

        ## Converting TransactionDate to datetime LIKE TO PYTHON DATE OBJECT
        data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

        ## Creating time features
        data['Hour'] = data['TransactionDate'].dt.hour
        data['DayOfWeek'] = data['TransactionDate'].dt.dayofweek
        data['Month'] = data['TransactionDate'].dt.month

        ## Encoding TransactionType INTO 0s and 1s (purchase/refund)
        data['TransactionType_encoded'] = self.label_encoder.fit_transform(data['TransactionType'])

        ## Encoding Location (one simple way)
        data['Location_encoded'] = self.label_encoder.fit_transform(data['Location'])

        ## Selecting features for modeling like removing unnecessary like customer name
        feature_columns = ['Amount', 'Hour', 'DayOfWeek', 'Month',
                           'TransactionType_encoded', 'Location_encoded']

        X = data[feature_columns] ## features
        y = data['IsFraud'] ## target

        print(f"Created {len(feature_columns)} features")
        print("Features:", feature_columns)

        return X, y

    def split_data(self, X, y, test_size=0.2):
        """splitting data into train and test"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


        print(f"\nTrain set: {len(X_train)} transactions")
        print(f"Test set: {len(X_test)} transactions")
        print(f"Fraud in train: {y_train.sum()} ({y_train.mean() * 100:.2f}%)")
        print(f"Fraud in test: {y_test.sum()} ({y_test.mean() * 100:.2f}%)")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Scaling the features (Amount is usually the only one needing scaling)"""
        self.scaler = RobustScaler()  ## Good for Amount which might have outliers

        ## Scaling only numeric columns (Amount in this case)
        cols_to_scale = ['Amount']
        other_cols = [col for col in X_train.columns if col not in cols_to_scale]

        if cols_to_scale:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()


             ## calculating median and IQR from train set and applying same to test set
            ## NO data leakage meaning model won't know how will be future data so just learning from train
            X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
            X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])

            print(f"Scaled columns: {cols_to_scale}")
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        return X_train_scaled, X_test_scaled

    def save_preprocessor(self, filename='preprocessor.pkl'):
        """Saving the preprocessor"""
        save_path = PROJECT_ROOT / 'models' / 'artifacts' / 'preprocessing.pkl'
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, save_path)
        print(f"Preprocessor saved to {save_path}")

    def load_preprocessor(self, filename='preprocessor.pkl'):
        """Loading the preprocessor"""
        data = joblib.load(filename)
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        print(f"Preprocessor loaded from {filename}")
        return self



if __name__ == "__main__":
    ## Loading  data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} transactions")

    ## Creating object for preprocessor
    prep = SimpleFraudPreprocessor()

    ## Preparing features
    print("\nPreparing features...")
    X, y = prep.prepare_features(df)

    ## Splitting data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = prep.split_data(X, y)

    ## feature scale
    print("\nScaling features...")
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    ## Saving preprocessor
    print("\nSaving preprocessor...")
    prep.save_preprocessor()

    print("\n Done! Data is ready for modeling.")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")