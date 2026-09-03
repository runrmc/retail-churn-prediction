"""Load and clean the Online Retail II dataset."""

import os
from typing import Tuple

import kagglehub
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def load_raw_data() -> pd.DataFrame:
    """Download (or use cached) dataset via kagglehub, return as a raw DataFrame."""
    path = kagglehub.dataset_download('mashlyn/online-retail-II-uci')
    df = pd.read_csv(os.path.join(path, 'online_retail_II.csv'))
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with missing Customer ID, fix dtype, split purchases from cancellations."""
    df_clean = df.dropna(subset=['Customer ID']).copy()
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # split cancellations out from actual purchases
    is_cancellation = df_clean['Invoice'].astype(str).str.startswith('C')
    cancellations = df_clean[is_cancellation].copy()
    purchases = df_clean[~is_cancellation].copy()

    return purchases, cancellations


def load_and_clean_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function: load raw data and return cleaned purchases + cancellations."""
    df = load_raw_data()
    purchases, cancellations = clean_data(df)
    return purchases, cancellations


if __name__ == '__main__':
    purchases, cancellations = load_and_clean_data()
    print(f'Purchases: {len(purchases):,} rows')
    print(f'Cancellations: {len(cancellations):,} rows')
    print(f'Unique customers: {purchases["Customer ID"].nunique():,}')
