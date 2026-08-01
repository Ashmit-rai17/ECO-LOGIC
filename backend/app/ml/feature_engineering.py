from __future__ import annotations

import numpy as np
import pandas as pd


LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 168]


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["hour"] = output.index.hour
    output["dayofweek"] = output.index.dayofweek
    output["dayofyear"] = output.index.dayofyear
    output["month"] = output.index.month
    output["quarter"] = output.index.quarter
    output["year"] = output.index.year
    output["is_weekend"] = (output.index.dayofweek >= 5).astype(int)
    return output


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for lag in LAGS:
        output[f"lag_{lag}"] = output["demand_mw"].shift(lag)
    return output


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    shifted = output["demand_mw"].shift(1)
    output["rolling_mean_24"] = shifted.rolling(24).mean()
    output["rolling_std_24"] = shifted.rolling(24).std()
    output["rolling_min_24"] = shifted.rolling(24).min()
    output["rolling_max_24"] = shifted.rolling(24).max()
    output["rolling_mean_168"] = shifted.rolling(168).mean()
    output["rolling_std_168"] = shifted.rolling(168).std()
    return output


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["hour_sin"] = np.sin(2 * np.pi * output["hour"] / 24)
    output["hour_cos"] = np.cos(2 * np.pi * output["hour"] / 24)
    output["month_sin"] = np.sin(2 * np.pi * output["month"] / 12)
    output["month_cos"] = np.cos(2 * np.pi * output["month"] / 12)
    output["dayofweek_sin"] = np.sin(2 * np.pi * output["dayofweek"] / 7)
    output["dayofweek_cos"] = np.cos(2 * np.pi * output["dayofweek"] / 7)
    return output


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    output = add_datetime_features(df)
    output = add_lag_features(output)
    output = add_rolling_features(output)
    output = add_cyclical_features(output)
    return output.dropna()


def feature_catalog() -> list[dict]:
    return [
        {
            "group": "Datetime Features",
            "features": ["hour", "dayofweek", "dayofyear", "month", "quarter", "year"],
            "description": "Calendar position gives the model recurring demand structure by hour, day, season, and year.",
        },
        {
            "group": "Lag Features",
            "features": [f"lag_{lag}" for lag in LAGS],
            "description": "Prior demand values preserve short-term inertia, daily repetition, and weekly repetition.",
        },
        {
            "group": "Rolling Features",
            "features": [
                "rolling_mean_24",
                "rolling_std_24",
                "rolling_min_24",
                "rolling_max_24",
                "rolling_mean_168",
                "rolling_std_168",
            ],
            "description": "Shifted rolling windows summarize the recent day and week without leaking the current target.",
        },
        {
            "group": "Calendar Features",
            "features": ["is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"],
            "description": "Weekend and cyclical encodings help tree splits handle periodic calendar boundaries.",
        },
    ]
