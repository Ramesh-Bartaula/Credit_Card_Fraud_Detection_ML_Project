import joblib
import pandas as pd
import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from pathlib import Path
import matplotlib.pyplot as plt

## Importing preprocessor class
from src.data.preprocessing import SimpleFraudPreprocessor

## Paths setup
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_yaml(file_name):
    config_path = PROJECT_ROOT / 'configs' / file_name
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


paths_config = load_yaml('config.yaml')
params_config = load_yaml('model_params.yaml')

## Pulling paths from config.yaml
DATA_PATH = PROJECT_ROOT / paths_config['paths']['raw_data']
MODEL_SAVE_PATH = PROJECT_ROOT / paths_config['paths']['model_path']


def find_best_threshold(model, X_test_scaled, y_test, min_recall=0.60):
    """
    Find the threshold that maximises F1 score on the fraud class,
    while guaranteeing a minimum recall of min_recall.

    min_recall=0.60 means we always catch at least 60% of fraud,
    even if it costs some precision (i.e. more false alarms).

    Lowering min_recall → fewer false alarms but more missed fraud.
    Raising min_recall  → catches more fraud but more false alarms.
    """
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    best_threshold = 0.5
    best_f1 = 0

    print("\n--- Threshold Tuning ---")
    print(f"{'Threshold':<12} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'False Alarms':<12} | {'Fraud Caught'}")
    print("-" * 75)

    for threshold in np.arange(0.10, 0.70, 0.05):
        preds = (y_proba >= threshold).astype(int)
        recall      = recall_score(y_test, preds, zero_division=0)
        prec        = precision_score(y_test, preds, zero_division=0)
        f1          = f1_score(y_test, preds, zero_division=0)
        cm          = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        ## Mark the best candidate — must meet minimum recall threshold
        meets_recall = recall >= min_recall
        is_best = (f1 > best_f1) and meets_recall
        marker = " <-- best" if is_best else ("" if meets_recall else " (recall too low)")

        print(f"{threshold:<12.2f} | {recall:<8.2f} | {prec:<10.4f} | {f1:<8.4f} | {fp:<12} | {tp}{marker}")

        if is_best:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold: {best_threshold:.2f} | F1: {best_f1:.4f} | Min recall enforced: {min_recall}")
    return best_threshold


def train_fraud_model():

    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    print("--- Loading Data ---")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['IsFraud'].sum()} / {len(df)} ({df['IsFraud'].mean()*100:.2f}%)")

    # -------------------------------------------------------
    # 2. PREPROCESS
    # -------------------------------------------------------
    print("\n--- Preprocessing ---")
    prep = SimpleFraudPreprocessor()
    X, y = prep.prepare_features(df, fit_encoders=True)
    X_train, X_test, y_train, y_test = prep.split_data(X, y)
    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)

    # -------------------------------------------------------
    # 3. SMOTE — oversample fraud cases in TRAINING data only
    #    Never apply SMOTE to test data — that would leak
    #    synthetic samples into evaluation and inflate scores
    # -------------------------------------------------------
    print("\n--- Applying SMOTE to Training Data ---")
    print(f"Before SMOTE — Normal: {(y_train == 0).sum()} | Fraud: {(y_train == 1).sum()}")

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    print(f"After SMOTE  — Normal: {(y_train_res == 0).sum()} | Fraud: {(y_train_res == 1).sum()}")

    # -------------------------------------------------------
    # 4. BASELINE MODEL (Logistic Regression)
    #    Used as a sanity check — if RF can't beat this,
    #    something is wrong with the features or training
    # -------------------------------------------------------
    print("\n--- Running Baseline Model ---")
    baseline = LogisticRegression(class_weight='balanced', max_iter=1000)
    baseline.fit(X_train_res, y_train_res)

    baseline_proba = baseline.predict_proba(X_test_scaled)[:, 1]

    ## Find best threshold for baseline too, with same min_recall rule
    baseline_best_threshold = find_best_threshold(baseline, X_test_scaled, y_test, min_recall=0.60)
    baseline_pred = (baseline_proba >= baseline_best_threshold).astype(int)

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_recall   = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1       = f1_score(y_test, baseline_pred, zero_division=0)
    baseline_auc      = roc_auc_score(y_test, baseline_proba)

    # -------------------------------------------------------
    # 5. RANDOM FOREST MODEL
    # -------------------------------------------------------
    rf_params = params_config['model_params']
    model = RandomForestClassifier(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        min_samples_split=rf_params['min_samples_split'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        max_features=rf_params['max_features'],
        class_weight=rf_params['class_weight'],
        random_state=rf_params['random_state'],
        n_jobs=rf_params['n_jobs']
    )

    print("\n--- Training Random Forest ---")
    model.fit(X_train_res, y_train_res)  # Train on SMOTE-resampled data

    # -------------------------------------------------------
    # 6. THRESHOLD TUNING
    #    Find the threshold that maximises F1 while keeping
    #    recall >= 60% — so we always catch most fraud cases
    # -------------------------------------------------------
    best_threshold = find_best_threshold(model, X_test_scaled, y_test, min_recall=0.60)

    # -------------------------------------------------------
    # 7. FINAL PREDICTIONS using best threshold
    # -------------------------------------------------------
    print("\n--- Evaluation ---")
    rf_proba = model.predict_proba(X_test_scaled)[:, 1]
    rf_pred  = (rf_proba >= best_threshold).astype(int)

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_recall   = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1       = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc      = roc_auc_score(y_test, rf_proba)

    # -------------------------------------------------------
    # 8. MODEL COMPARISON TABLE
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("                    MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<12} | {'Baseline (LR)':>15} | {'Random Forest':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<12} | {baseline_accuracy:>15.2f} | {rf_accuracy:>15.2f}")
    print(f"{'Recall':<12} | {baseline_recall:>15.2f} | {rf_recall:>15.2f}")
    print(f"{'F1 Score':<12} | {baseline_f1:>15.2f} | {rf_f1:>15.2f}")
    print(f"{'AUC-ROC':<12} | {baseline_auc:>15.2f} | {rf_auc:>15.2f}")
    print("=" * 60)

    # -------------------------------------------------------
    # 9. DETAILED CLASSIFICATION REPORT
    # -------------------------------------------------------
    print(f"\nClassification Report — Random Forest (threshold={best_threshold:.2f})")
    print(classification_report(y_test, rf_pred, target_names=['Normal', 'Fraud']))

    # -------------------------------------------------------
    # 10. CONFUSION MATRIX — with plain English breakdown
    # -------------------------------------------------------
    cmf = confusion_matrix(y_test, rf_pred)
    tn, fp, fn, tp = cmf.ravel()

    print("Confusion Matrix:")
    print(cmf)
    print(f"\nTrue Negatives  (Normal correctly identified): {tn}")
    print(f"False Positives (Normal flagged as Fraud):     {fp}  ← false alarms")
    print(f"False Negatives (Fraud missed):                {fn}  ← want this LOW")
    print(f"True Positives  (Fraud correctly caught):      {tp}  ← want this HIGH")
    print(f"\nFraud detection rate: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}%")
    print(f"False alarm rate:      {fp}/{fp+tn} = {fp/(fp+tn)*100:.2f}% of normal transactions")

    display = ConfusionMatrixDisplay(confusion_matrix=cmf, display_labels=['Normal', 'Fraud'])
    display.plot(cmap='Blues')
    plt.title(f'Random Forest — Threshold {best_threshold:.2f}')

    reports_dir = PROJECT_ROOT / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(reports_dir / 'confusion_matrix.png', bbox_inches='tight')
    print(f"\nConfusion matrix saved to {reports_dir / 'confusion_matrix.png'}")
    plt.show()

    # -------------------------------------------------------
    # 11. FEATURE IMPORTANCE — helps understand what the
    #     model is actually learning
    # -------------------------------------------------------
    feature_names = ['Amount', 'Hour', 'DayOfWeek', 'Month',
                     'TransactionType_encoded', 'Location_encoded']
    importances = model.feature_importances_

    print("\n--- Feature Importances ---")

    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = '█' * int(imp * 100)
        print(f"{name:<28} {imp:.4f}  {bar}")

    # -------------------------------------------------------
    # 12. SAVE MODEL, PREPROCESSOR, AND THRESHOLD
    # -------------------------------------------------------
    prep.save_preprocessor()

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    ## Save threshold alongside model — inference must use the same value
    threshold_path = MODEL_SAVE_PATH.parent / 'best_threshold.joblib'
    joblib.dump(best_threshold, threshold_path)
    print(f"Best threshold ({best_threshold:.2f}) saved to {threshold_path}")


if __name__ == "__main__":
    train_fraud_model()