"""
features.py

Feature engineering and preprocessing pipeline.
Returns X_train, X_test, y_train, y_test after preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(df, numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()
    # remove target if present
    for t in ["y","target"]:
        if t in numeric_features: numeric_features.remove(t) if t in numeric_features else None
        if t in categorical_features: categorical_features.remove(t) if t in categorical_features else None

    numeric_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    return preprocessor

def prepare_data(df, target_col="y", test_size=0.2, random_state=42):
    df = df.copy()
    # convert target to binary if needed
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'yes':1, 'no':0})
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
