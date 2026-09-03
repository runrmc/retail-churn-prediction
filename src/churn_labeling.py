"""Define churn labels based on a reference date and lookback/outcome windows."""

from typing import Tuple

import pandas as pd

from src.data_loader import load_and_clean_data


CHURN_WINDOW_DAYS = 180


def get_reference_date(purchases: pd.DataFrame, window_days: int = CHURN_WINDOW_DAYS) -> pd.Timestamp:
    """Return the latest reference date that leaves a full window_days outcome period."""
    last_date = purchases['InvoiceDate'].max()
    reference_date = last_date - pd.Timedelta(days=window_days)
    return reference_date


def split_feature_and_label_windows(purchases: pd.DataFrame, reference_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split purchases into a feature window (on/before reference date) and a label window (after)."""
    feature_window = purchases[purchases['InvoiceDate'] <= reference_date]
    label_window = purchases[purchases['InvoiceDate'] > reference_date]
    return feature_window, label_window


def build_churn_labels(purchases: pd.DataFrame, window_days: int = CHURN_WINDOW_DAYS) -> pd.DataFrame:
    """
    Build a churn label table.
    
    A customer is considered active if they purchased on/before the reference date.
    They are labeled churned=1 if they made no purchases in the window_days after it.
    """
    reference_date = get_reference_date(purchases, window_days)
    feature_window, label_window = split_feature_and_label_windows(purchases, reference_date)

    active_customers = feature_window['Customer ID'].unique()
    returning_customers = label_window['Customer ID'].unique()

    churn_labels = pd.DataFrame({'Customer ID': active_customers})
    churn_labels['churned'] = (~churn_labels['Customer ID'].isin(returning_customers)).astype(int)

    return churn_labels


if __name__ == '__main__':
    purchases, cancellations = load_and_clean_data()
    reference_date = get_reference_date(purchases)
    churn_labels = build_churn_labels(purchases)

    print(f'Reference date: {reference_date.date()}')
    print(f'Active customers: {len(churn_labels):,}')
    print(f'Churn rate: {churn_labels["churned"].mean():.2%}')
    print(churn_labels['churned'].value_counts())
