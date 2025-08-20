"""
data_loader.py

Functions:
- load_data(path): load CSV into pandas DataFrame
- basic_cleaning(df): basic cleaning and type adjustments
"""

import pandas as pd

def load_data(path="data/bank.csv"):
    """
    Loads dataset from path. Expects a CSV with ; separator or , ; try both.
    """
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path, sep=',')
    return df

def basic_cleaning(df):
    # Lowercase column names
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Strip string columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    return df
