"""Streamlit dashboard for the retail churn prediction project."""

import sys
import os

from typing import Dict, Tuple
import pandas as pd

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_clean_data
from src.features import build_features
from src.model import split_data, load_model, MODEL_PATH, WEIGHTED_MODEL_PATH
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier

st.set_page_config(page_title='Retail Churn Prediction', layout='wide')

st.title('Retail Customer Churn Prediction')

# load data and models
@st.cache_data
def get_engineered_data() -> pd.DataFrame:
    """Load and clean raw data, then build the full engineered feature table."""
    purchases, cancellations = load_and_clean_data()
    engineered_df = build_features(purchases, cancellations)
    return engineered_df

@st.cache_resource
def get_models() -> Tuple[XGBClassifier, XGBClassifier]:
    """Load the standard and value-weighted trained models from disk."""
    standard_model = load_model(MODEL_PATH)
    weighted_model = load_model(WEIGHTED_MODEL_PATH)
    return standard_model, weighted_model

engineered_df = get_engineered_data()
standard_model, weighted_model = get_models()

# compute metrics for both models
X_train, X_test, y_train, y_test = split_data(engineered_df)

def get_metrics(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute ROC-AUC, recall, and precision for a trained model on the test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
    }

standard_metrics = get_metrics(standard_model, X_test, y_test)
weighted_metrics = get_metrics(weighted_model, X_test, y_test)

# display
st.header('Model Performance')

model_choice = st.radio(
    label='Select model to view:',
    options=['Standard', 'Value-Weighted'],
    horizontal=True
)

active_metrics = standard_metrics if model_choice == 'Standard' else weighted_metrics

if model_choice == 'Value-Weighted':
    st.caption('Trained with monetary value as sample weight - prioritizes catching high-spend churners, at a small cost to overall recall.')

col1, col2, col3 = st.columns(3)
col1.metric('ROC-AUC', f'{active_metrics["roc_auc"]:.3f}')
col2.metric('Recall', f'{active_metrics["recall"]:.3f}')
col3.metric('Precision', f'{active_metrics["precision"]:.3f}')

st.header('Visualizations')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Feature Importance')
    st.image('reports/feature_importance.png')

    st.subheader('RFM Distributions by Churn Status')
    st.image('reports/rfm_distribution_by_churn.png')

with col2:
    st.subheader('Revenue at Risk: Missed vs. Caught Churners')
    st.image('reports/revenue_at_risk_gap.png')
