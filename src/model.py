"""Train, evaluate, and persist the churn prediction model."""

import argparse
import os
from typing import Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

from src.data_loader import load_and_clean_data
from src.features import build_features

FEATURE_COLS = ['recency_days', 'frequency', 'monetary', 'return_rate']
MODEL_PATH = '../models/xgb_churn_model.joblib'
WEIGHTED_MODEL_PATH = '../models/xgb_churn_model_weighted.joblib'

# best hyperparamers found via grid search (see notebooks/01_eda.ipynb)
BEST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 2,
    'min_child_weight': 1,
    'n_estimators': 50,
}


def split_data(
    engineered_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split engineered into train/test sets, stratified on churn label."""
    X = engineered_df[FEATURE_COLS]
    y = engineered_df['churned']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series = None) -> XGBClassifier:
    """Train XGBoost using the pre-tuned hyperparamers. Optionally pass sample_weight for value-weighted training."""
    model = XGBClassifier(**BEST_PARAMS, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print confusion matrix, classification report, and ROC-AUC for a trained model."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(f'ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}')


def save_model(model: XGBClassifier, path: str = MODEL_PATH) -> None:
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f'Model saved to {path}')


def load_model(path: str = MODEL_PATH) -> XGBClassifier:
    """Load a previously saved model from disk."""
    return joblib.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the churn prediction model.')
    parser.add_argument(
        '--weighted',
        action='store_true',
        help='Train using monetary value as a sample weight, prioritizing high-value customers.'
    )
    args = parser.parse_args()

    purchases, cancellations = load_and_clean_data()
    engineered_df = build_features(purchases, cancellations)

    X_train, X_test, y_train, y_test = split_data(engineered_df)

    if args.weighted:
        print('Training value-weighted model (sample_weight=monetary)...')
        train_monetary = engineered_df.loc[X_train.index, 'monetary']
        model = train_model(X_train, y_train, sample_weight=train_monetary)
        save_path = WEIGHTED_MODEL_PATH
    else:
        print('Training standard model...')
        model = train_model(X_train, y_train)
        save_path = MODEL_PATH

    print('\nEvaluating on test set:')
    evaluate_model(model, X_test, y_test)

    save_model(model, path=save_path)
