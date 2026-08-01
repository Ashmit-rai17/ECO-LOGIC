from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sample_timeseries(df: pd.DataFrame, max_points: int) -> list[dict]:
    if len(df) > max_points:
        stride = math.ceil(len(df) / max_points)
        df = df.iloc[::stride]
    return [{"timestamp": index.isoformat(), **{key: float(value) for key, value in row.items()}} for index, row in df.iterrows()]


def analysis_payload(df: pd.DataFrame, max_points: int) -> dict:
    rolling = pd.DataFrame(
        {
            "demand_mw": df["demand_mw"],
            "rolling_mean": df["demand_mw"].rolling(24 * 7).mean(),
            "rolling_std": df["demand_mw"].rolling(24 * 7).std(),
        }
    ).dropna()
    daily = df[["demand_mw"]].resample("D").mean()
    heatmap = (
        df.assign(dayofweek=df.index.dayofweek, hour=df.index.hour)
        .pivot_table(values="demand_mw", index="dayofweek", columns="hour", aggfunc="mean")
        .replace({np.nan: None})
    )
    return {
        "historical": sample_timeseries(df[["demand_mw"]], max_points),
        "dailyDemand": sample_timeseries(daily, max_points),
        "rolling": sample_timeseries(rolling, max_points),
        "distribution": [float(value) for value in df["demand_mw"].dropna().sample(min(3500, len(df)), random_state=42)],
        "profiles": {
            "hourly": [{"hour": int(k), "demand_mw": float(v)} for k, v in df.groupby(df.index.hour)["demand_mw"].mean().items()],
            "weekday": [{"dayofweek": int(k), "demand_mw": float(v)} for k, v in df.groupby(df.index.dayofweek)["demand_mw"].mean().items()],
            "monthly": [{"month": int(k), "demand_mw": float(v)} for k, v in df.groupby(df.index.month)["demand_mw"].mean().items()],
            "yearly": [{"year": int(k), "demand_mw": float(v)} for k, v in df.groupby(df.index.year)["demand_mw"].mean().items()],
            "weekend": [
                {"label": "Weekday", "demand_mw": float(df.loc[df.index.dayofweek < 5, "demand_mw"].mean())},
                {"label": "Weekend", "demand_mw": float(df.loc[df.index.dayofweek >= 5, "demand_mw"].mean())},
            ],
        },
        "calendarHeatmap": {
            "z": heatmap.values.tolist(),
            "x": [int(col) for col in heatmap.columns],
            "y": [int(row) for row in heatmap.index],
        },
    }


def forecast_series(dashboard: pd.DataFrame, max_points: int) -> dict:
    window = dashboard[["actual", "predicted", "residual", "absolute_error"]]
    return {
        "full": sample_timeseries(window, max_points),
        "zoom": sample_timeseries(window.head(24 * 14), max_points),
        "worstPredictions": [
            {"timestamp": index.isoformat(), **{key: float(value) for key, value in row.items()}}
            for index, row in dashboard.sort_values("absolute_error", ascending=False).head(20).iterrows()
        ],
    }


def error_heatmap(dashboard: pd.DataFrame) -> dict:
    heatmap = dashboard.pivot_table(values="absolute_error", index="dayofweek", columns="hour", aggfunc="mean")
    return {
        "z": heatmap.values.tolist(),
        "x": [int(col) for col in heatmap.columns],
        "y": [int(row) for row in heatmap.index],
    }
