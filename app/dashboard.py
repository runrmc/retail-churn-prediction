"""Streamlit dashboard for the retail churn prediction project."""

import sys
import os

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_clean_data
from src.features import build_features
from src.model import split_data, load_model, MODEL_PATH, WEIGHTED_MODEL_PATH
from sklearn.metrics import roc_auc_score, recall_score, precision_score

st.set_page_config(page_title='Retail Churn Prediction', layout='wide')

st.title('Retail Customer Churn Prediction')

# load data and models
@st.cache_data
def get_engineered_data():
    purchases, cancellations = load_and_clean_data()
    engineered_df = build_features(purchases, cancellations)
    return engineered_df

@st.cache_resource
def get_models():
    standard_model = load_model(MODEL_PATH)
    weighted_model = load_model(WEIGHTED_MODEL_PATH)
    return standard_model, weighted_model

engineered_df = get_engineered_data()
standard_model, weighted_model = get_models()

st.write('Dashboard is working - data and models loaded.')

