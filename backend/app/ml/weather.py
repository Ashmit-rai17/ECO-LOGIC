"""
Weather data integration for ECO-LOGIC.

Fetches historical hourly weather from Open-Meteo (free, no API key required).
Provides dataset-specific geographic configurations and weather feature engineering
including HDD/CDD (Heating/Cooling Degree Hours).

Usage:
    from app.ml.weather import fetch_weather, build_weather_features, DATASET_LOCATIONS

    weather_df = fetch_weather("AEP", start_date, end_date)
    model_df_with_weather = build_weather_features(model_df, weather_df, t_base_cooling=18.3, t_base_heating=18.3)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Dataset geographic configuration
# ---------------------------------------------------------------------------
# PJM regions — lat/lon represent the approximate center of each utility's
# service territory.  These are NOT exact points; the weather is a regional
# approximation suitable for demand-level modeling.

DATASET_LOCATIONS: dict[str, dict] = {
    "AEP": {
        "latitude": 39.96,
        "longitude": -82.99,
        "timezone": "America/New_York",
        "region": "American Electric Power (Ohio)",
        "t_base_cooling": 18.3,  # 65°F
        "t_base_heating": 18.3,  # 65°F
    },
    "COMED": {
        "latitude": 41.88,
        "longitude": -87.63,
        "timezone": "America/Chicago",
        "region": "Commonwealth Edison (Chicago)",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "DAYTON": {
        "latitude": 39.76,
        "longitude": -84.19,
        "timezone": "America/New_York",
        "region": "Dayton Power & Light (Ohio)",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "DEOK": {
        "latitude": 39.96,
        "longitude": -82.99,
        "timezone": "America/New_York",
        "region": "Duke Energy Ohio/Kentucky",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "DOM": {
        "latitude": 37.54,
        "longitude": -77.43,
        "timezone": "America/New_York",
        "region": "Dominion Energy (Virginia)",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "DUQ": {
        "latitude": 40.44,
        "longitude": -79.99,
        "timezone": "America/New_York",
        "region": "Duquesne Light (Pittsburgh)",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "EKPC": {
        "latitude": 38.25,
        "longitude": -85.76,
        "timezone": "America/New_York",
        "region": "East Kentucky Power Cooperative",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "FE": {
        "latitude": 40.34,
        "longitude": -75.93,
        "timezone": "America/New_York",
        "region": "FirstEnergy (Pennsylvania)",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "NI": {
        "latitude": 41.26,
        "longitude": -81.52,
        "timezone": "America/New_York",
        "region": "Northern Indiana Public Service",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
    "PJME": {
        "latitude": 40.0,
        "longitude": -77.0,
        "timezone": "America/New_York",
        "region": "PJM Eastern Interconnection",
        "t_base_cooling": 18.3,
        "t_base_heating": 18.3,
    },
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Weather fetching
# ---------------------------------------------------------------------------

def fetch_weather(
    dataset_key: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch historical hourly weather for a dataset from Open-Meteo.

    Returns a DataFrame with columns:
        timestamp (DatetimeIndex), temperature_2m, relative_humidity_2m,
        wind_speed_10m, precipitation, cloud_cover

    Results are cached to disk if cache_dir is provided.
    """
    key = dataset_key.upper()
    if key not in DATASET_LOCATIONS:
        raise KeyError(f"No location configured for dataset '{key}'")

    loc = DATASET_LOCATIONS[key]

    # Check cache
    cache_path = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}_weather_{start_date}_{end_date}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            return _dict_to_weather_df(data)

    # Fetch from Open-Meteo
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
        ],
        "timezone": loc["timezone"],
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Cache to disk
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(data, f)

    return _dict_to_weather_df(data)


def _dict_to_weather_df(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a weather DataFrame."""
    hourly = data.get("hourly", {})
    timestamps = hourly.get("time", [])

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "wind_speed_10m": hourly.get("wind_speed_10m", []),
        "precipitation": hourly.get("precipitation", []),
        "cloud_cover": hourly.get("cloud_cover", []),
    })

    df = df.set_index("timestamp")
    df.index = df.index.tz_localize(None)  # Remove timezone for demand alignment
    return df


# ---------------------------------------------------------------------------
# Weather feature engineering
# ---------------------------------------------------------------------------

def compute_hdd_cdd(
    temperature: pd.Series,
    t_base_cooling: float = 18.3,
    t_base_heating: float = 18.3,
) -> pd.DataFrame:
    """
    Compute Heating Degree Hours (HDH) and Cooling Degree Hours (CDH).

    HDD = max(T_base - temperature, 0)   → how much heating is needed
    CDD = max(temperature - T_base, 0)    → how much cooling is needed

    T_base = 18.3°C (65°F) is the standard reference temperature for
    electricity demand modeling (ASHRAE standard).
    """
    return pd.DataFrame({
        "hdd": np.maximum(t_base_heating - temperature, 0),
        "cdd": np.maximum(temperature - t_base_cooling, 0),
    }, index=temperature.index)


def build_weather_features(
    model_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    t_base_cooling: float = 18.3,
    t_base_heating: float = 18.3,
) -> pd.DataFrame:
    """
    Merge weather data into the model DataFrame and engineer weather features.

    Features added:
        temperature_2m, relative_humidity_2m, wind_speed_10m,
        precipitation, cloud_cover,
        hdd, cdd,
        temp_lag_24 (temperature 24h ago),
        temp_change_1h (hourly temperature change),
        temp_squared (non-linear temperature effect)

    Timestamp alignment:
        Weather timestamps are aligned to the demand DataFrame's DatetimeIndex
        using merge_asof (backward) to handle any small mismatches.
    """
    result = model_df.copy()

    # Align weather to demand timestamps using merge_asof
    demand_ts = pd.DataFrame({"timestamp": result.index}).sort_values("timestamp")
    weather_ts = weather_df.reset_index().rename(columns={"index": "timestamp"}).sort_values("timestamp")

    merged = pd.merge_asof(
        demand_ts, weather_ts, on="timestamp", direction="backward"
    )
    merged = merged.set_index("timestamp")

    # Add raw weather columns
    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloud_cover"]:
        if col in merged.columns:
            result[col] = merged[col].values

    # HDD / CDD
    temp = result["temperature_2m"]
    hdd_cdd = compute_hdd_cdd(temp, t_base_cooling, t_base_heating)
    result["hdd"] = hdd_cdd["hdd"]
    result["cdd"] = hdd_cdd["cdd"]

    # Temperature derivatives
    result["temp_lag_24"] = temp.shift(24)
    result["temp_change_1h"] = temp.diff(1)
    result["temp_squared"] = temp ** 2

    return result


def weather_feature_catalog() -> list[dict]:
    """Extend the feature catalog with weather feature descriptions."""
    return [
        {
            "group": "Weather Features",
            "features": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "precipitation",
                "cloud_cover",
            ],
            "description": (
                "Hourly weather observations from Open-Meteo reanalysis data, "
                "aligned to each dataset's geographic region."
            ),
        },
        {
            "group": "Derived Weather Features",
            "features": [
                "hdd",
                "cdd",
                "temp_lag_24",
                "temp_change_1h",
                "temp_squared",
            ],
            "description": (
                "Heating/Cooling Degree Hours quantify thermal demand pressure. "
                "Temperature lag and change capture the rate of weather shifts "
                "that drive ramp-up/ramp-down in electricity consumption."
            ),
        },
    ]
