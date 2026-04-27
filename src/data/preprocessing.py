import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

# Importing our new feature logic
from src.features.build_features import FraudFeatureBuilder

PROJECT_ROOT = Path(__file__).parent.parent.parent


class SimpleFraudPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.type_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.feature_builder = FraudFeatureBuilder()
        self._encoders_fitted = False

    def prepare_features(self, df, fit_encoders=True):
        data = self.feature_builder.add_time_features(df)

        if fit_encoders:
            self.feature_builder.fit_merchant_stats(data)
            data['TransactionType_encoded'] = self.type_encoder.fit_transform(data['TransactionType'])
            data['Location_encoded'] = self.location_encoder.fit_transform(data['Location'])
            self._encoders_fitted = True
        else:
            data['TransactionType_encoded'] = self.type_encoder.transform(data['TransactionType'])
            data['Location_encoded'] = self.location_encoder.transform(data['Location'])

        data = self.feature_builder.add_merchant_features(data)

        feature_columns = [
            'Amount', 'Hour', 'DayOfWeek', 'Month', 'TransactionType_encoded',
            'Location_encoded', 'Is_night', 'Is_weekend', 'Merchant_fraud_rate',
            'Merchant_tx_count', 'Merchant_avg_amount', 'Amount_vs_merchant_avg'
        ]
        return data[feature_columns], data['IsFraud']

    def split_data(self, X, y, test_size=0.3):
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def scale_features(self, X_train, X_test):
        cols_to_scale = ['Amount', 'Merchant_avg_amount', 'Amount_vs_merchant_avg']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[cols_to_scale] = self.scaler.fit_transform(X_train[cols_to_scale])
        X_test_scaled[cols_to_scale] = self.scaler.transform(X_test[cols_to_scale])
        return X_train_scaled, X_test_scaled

    def save_preprocessor(self):
        save_path = PROJECT_ROOT / 'models' / 'artifacts' / 'preprocessor.joblib'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts = {
            'scaler': self.scaler,
            'type_encoder': self.type_encoder,
            'location_encoder': self.location_encoder,
            'feature_builder': self.feature_builder
        }
        joblib.dump(artifacts, save_path)
        print(f"✓ Preprocessor saved to {save_path}")