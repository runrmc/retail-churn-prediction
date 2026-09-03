"""Build RFM (Recency, Frequency, Monetary) and return-rate features per customer."""

import pandas as pd

from src.data_loader import load_and_clean_data
from src.churn_labeling import get_reference_date, split_feature_and_label_windows, build_churn_labels


def compute_recency(feature_window: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Days since each customer's last purchase, as of the reference date."""
    recency = feature_window.groupby('Customer ID')['InvoiceDate'].max().reset_index()
    recency.columns = ['Customer ID', 'last_purchase_date']
    recency['recency_days'] = (reference_date - recency['last_purchase_date']).dt.days
    return recency[['Customer ID', 'recency_days']]


def compute_frequency(feature_window: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct purchase invoices per customer."""
    frequency = feature_window.groupby('Customer ID')['Invoice'].nunique().reset_index()
    frequency.columns = ['Customer ID', 'frequency']
    return frequency


def compute_monetary(feature_window: pd.DataFrame) -> pd.DataFrame:
    """Total amount spent per customer."""
    feature_window = feature_window.copy()
    feature_window['line_total'] = feature_window['Quantity'] * feature_window['Price']
    monetary = feature_window.groupby('Customer ID')['line_total'].sum().reset_index()
    monetary.columns = ['Customer ID', 'monetary']
    return monetary


def compute_return_rate(cancellations: pd.DataFrame, frequency: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Cancellations as a share of total orders, per customer."""
    cancellations_pre = cancellations[cancellations['InvoiceDate'] <= reference_date]
    returns = cancellations_pre.groupby('Customer ID')['Invoice'].nunique().reset_index()
    returns.columns = ['Customer ID', 'return_count']

    result = frequency.merge(returns, on='Customer ID', how='left')
    result['return_count'] = result['return_count'].fillna(0)
    result['return_rate'] = result['return_count'] / (result['frequency'] + result['return_count'])

    return result[['Customer ID', 'return_count', 'return_rate']]


def build_features(purchases: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    """Build the full engineered feature table, merged with churn labels."""
    reference_date = get_reference_date(purchases)
    feature_window, _ = split_feature_and_label_windows(purchases, reference_date)

    recency = compute_recency(feature_window, reference_date)
    frequency = compute_frequency(feature_window)
    monetary = compute_monetary(feature_window)
    returns = compute_return_rate(cancellations, frequency, reference_date)
    churn_labels = build_churn_labels(purchases)

    engineered_df = (
        recency
        .merge(frequency, on='Customer ID')
        .merge(monetary, on='Customer ID')
        .merge(returns[['Customer ID', 'return_rate']], on='Customer ID')
        .merge(churn_labels, on='Customer ID')
    )

    return engineered_df


if __name__ == '__main__':
    purchases, cancellations = load_and_clean_data()
    engineered_df = build_features(purchases, cancellations)

    print(f'Feature table shape: {engineered_df.shape}')
    print(engineered_df.head())
    print(f'\nChurn rate: {engineered_df["churned"].mean():.2%}')
