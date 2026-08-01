from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf


def stationarity(df: pd.DataFrame) -> dict:
    series = df["demand_mw"].dropna()
    result = adfuller(series)
    p_value = float(result[1])
    return {
        "adfStatistic": float(result[0]),
        "pValue": p_value,
        "criticalValues": {key: float(value) for key, value in result[4].items()},
        "interpretation": "Likely stationary at the 5% level." if p_value < 0.05 else "Likely non-stationary; trend or seasonal structure remains.",
    }


def correlation_summary(feature_df: pd.DataFrame) -> list[dict]:
    corr = feature_df.corr(numeric_only=True)["demand_mw"].drop("demand_mw").sort_values(key=lambda values: values.abs(), ascending=False)
    return [{"feature": key, "correlation": float(value)} for key, value in corr.head(15).items()]


def autocorrelation(series: pd.Series, nlags: int = 48) -> dict:
    clean = series.dropna()
    return {
        "acf": [{"lag": lag, "value": float(value)} for lag, value in enumerate(acf(clean, nlags=nlags, fft=True))],
        "pacf": [{"lag": lag, "value": float(value)} for lag, value in enumerate(pacf(clean, nlags=nlags, method="ywm"))],
    }


def seasonality_insights(df: pd.DataFrame) -> list[str]:
    hourly_peak = int(df.groupby(df.index.hour)["demand_mw"].mean().idxmax())
    monthly_peak = int(df.groupby(df.index.month)["demand_mw"].mean().idxmax())
    weekend_mean = float(df.loc[df.index.dayofweek >= 5, "demand_mw"].mean())
    weekday_mean = float(df.loc[df.index.dayofweek < 5, "demand_mw"].mean())
    weekend_delta = ((weekend_mean - weekday_mean) / weekday_mean) * 100
    return [
        f"Average demand peaks around hour {hourly_peak:02d}:00.",
        f"Monthly demand is highest in month {monthly_peak}.",
        f"Weekend demand is {abs(weekend_delta):.1f}% {'higher' if weekend_delta >= 0 else 'lower'} than weekday demand.",
    ]
