# misc.py
from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.base import RegressorMixin

DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
CACHE_FILE = "boston_housing_cached.csv"
RANDOM_STATE = 42


def load_data(cache: bool = True) -> pd.DataFrame:
    """
    Load Boston Housing dataset manually as per assignment instructions.
    If cache is True, save a local CSV the first time and reuse it later.
    """
    if cache and os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)

    try:
        raw_df = pd.read_csv(DATA_URL, sep="\s+", skiprows=22, header=None, engine="python")
    except Exception as e:
        raise RuntimeError(
            "Failed to download dataset. Ensure internet access or place 'boston_housing_cached.csv' in repo root."
        ) from e

    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target

    if cache:
        df.to_csv(CACHE_FILE, index=False)

    return df


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    X = df.drop(columns=["MEDV"]).values
    y = df["MEDV"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(model: RegressorMixin, X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray, label: str = "") -> float:
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    prefix = f" ({label})" if label else ""
    print(f"[RESULT] Test MSE{prefix}: {mse:.4f}")
    return mse
