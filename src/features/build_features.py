import pandas as pd

class FraudFeatureBuilder:
    def __init__(self):
        self.merchant_fraud_rate = {}
        self.merchant_tx_count = {}
        self.merchant_avg_amount = {}
        self.global_fraud_rate = 0.0
        self.global_avg_amount = 0.0

    def add_time_features(self, df):
        df = df.copy()
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['Hour'] = df['TransactionDate'].dt.hour
        df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
        df['Month'] = df['TransactionDate'].dt.month
        df['Is_night'] = df['Hour'].apply(lambda x: 1 if (x >= 23 or x <= 5) else 0)
        df['Is_weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        return df

    def fit_merchant_stats(self, df):
        self.merchant_fraud_rate = df.groupby('MerchantID')['IsFraud'].mean().to_dict()
        self.merchant_tx_count = df.groupby('MerchantID')['Amount'].count().to_dict()
        self.merchant_avg_amount = df.groupby('MerchantID')['Amount'].mean().to_dict()
        self.global_fraud_rate = df['IsFraud'].mean()
        self.global_avg_amount = df['Amount'].mean()

    def add_merchant_features(self, df):
        df['Merchant_fraud_rate'] = df['MerchantID'].map(self.merchant_fraud_rate).fillna(self.global_fraud_rate)
        df['Merchant_tx_count'] = df['MerchantID'].map(self.merchant_tx_count).fillna(1)
        df['Merchant_avg_amount'] = df['MerchantID'].map(self.merchant_avg_amount).fillna(self.global_avg_amount)
        df['Amount_vs_merchant_avg'] = df['Amount'] - df['Merchant_avg_amount']
        return df