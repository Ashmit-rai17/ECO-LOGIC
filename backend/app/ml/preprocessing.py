from __future__ import annotations

import pandas as pd


def prepare_time_index(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").set_index("Datetime")
    df.index.name = "Datetime"
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    return df


def overview(df: pd.DataFrame, dataset_key: str) -> dict:
    inferred_frequency = pd.infer_freq(df.index)
    duplicates = int(df.index.duplicated().sum())
    missing_values = int(df["demand_mw"].isna().sum())
    stats = df["demand_mw"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()

    return {
        "dataset": dataset_key,
        "timeRange": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
        },
        "observations": int(len(df)),
        "missingValues": missing_values,
        "duplicateTimestamps": duplicates,
        "samplingFrequency": inferred_frequency or "irregular/hourly expected",
        "summaryStatistics": {key: float(value) for key, value in stats.items()},
    }
