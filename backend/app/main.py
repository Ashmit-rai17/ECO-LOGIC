"""
ECO-LOGIC FastAPI Inference Server

Serves precomputed analytics, forecasts, and predictions from offline-trained
model artifacts.  No training happens here — all heavy ML is done by
``python -m training.train`` offline.

Artifacts are loaded lazily on first request and cached in memory.
The API performs lightweight data reshaping to match the frontend contract.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

logger = logging.getLogger("eco-logic")

settings = get_settings()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ECO-LOGIC Energy Intelligence API",
    version="2.0.0",
    description=(
        "Lightweight inference API serving precomputed analytics, forecasts, "
        "and model artifacts for the ECO-LOGIC energy demand dashboard."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------

ARTIFACTS_DIR: Path = settings.artifacts_dir


def _discover_datasets() -> list[dict]:
    """Scan the artifacts directory for valid dataset folders."""
    datasets: list[dict] = []
    if not ARTIFACTS_DIR.is_dir():
        logger.warning("Artifacts directory not found: %s", ARTIFACTS_DIR)
        return datasets

    for entry in sorted(ARTIFACTS_DIR.iterdir()):
        if entry.is_dir() and (entry / "analytics.json").exists():
            try:
                data = _load_raw(entry.name)
                meta = data.get("metadata", {})
                datasets.append(
                    {
                        "key": entry.name,
                        "label": meta.get("dataset_label", entry.name),
                    }
                )
            except Exception:
                datasets.append({"key": entry.name, "label": entry.name})
    return datasets


@lru_cache(maxsize=16)
def _load_raw(dataset_key: str) -> dict:
    """Load and cache the raw analytics artifact for a dataset."""
    path = ARTIFACTS_DIR / dataset_key.upper() / "analytics.json"
    if not path.is_file():
        raise KeyError(f"Dataset '{dataset_key}' not found — no artifact at {path}")
    with open(path) as f:
        return json.load(f)


def _build_frontend_payload(dataset_key: str) -> dict:
    """
    Transform raw artifacts into the exact shape the frontend expects.

    This is lightweight dict reshaping — no ML computation.
    """
    raw = _load_raw(dataset_key.upper())
    meta = raw.get("metadata", {})
    metrics = raw.get("metrics", {})
    overview = raw.get("overview", {})
    analysis = raw.get("analysis", {})
    statistics = raw.get("statistics", {})
    features = raw.get("features", {})
    forecast_series = raw.get("forecastSeries", {})
    errors = raw.get("errors", {})
    efficiency = raw.get("efficiency", {})
    summary = raw.get("summary", {})
    model_info = raw.get("modelInfo", {})
    model_comparison = raw.get("modelComparison", [])
    baseline_comparison = raw.get("baselineComparison", [])
    baseline_improvement = raw.get("baselineImprovement", 0)
    explainability = raw.get("explainability", {})

    # --- dataset ---
    dataset_obj = {
        "key": dataset_key.upper(),
        "label": meta.get("dataset_label", dataset_key.upper()),
    }

    # --- overview (already matches frontend shape) ---
    # Just ensure summaryStatistics has the right keys
    overview_obj = dict(overview)

    # --- statistics (add autocorrelation placeholder if missing) ---
    statistics_obj = dict(statistics)
    if "autocorrelation" not in statistics_obj:
        # Empty ACF/PACF — the frontend handles this gracefully
        statistics_obj["autocorrelation"] = {"acf": [], "pacf": []}

    # --- forecasting (assemble from multiple artifact fields) ---
    strategy = {
        "train": {
            "start": overview.get("timeRange", {}).get("start", ""),
            "end": "",
            "rows": 0,
        },
        "validation": {"start": "", "end": "", "rows": 0},
        "test": {
            "start": forecast_series.get("full", [{}])[0].get("timestamp", "") if forecast_series.get("full") else "",
            "end": forecast_series.get("full", [{}])[-1].get("timestamp", "") if forecast_series.get("full") else "",
            "rows": len(forecast_series.get("full", [])),
        },
    }

    # Get training curve from modelInfo
    training_curve = model_info.get("trainingCurve", {})
    forecasting_obj = {
        "strategy": strategy,
        "model": {
            "bestIteration": model_info.get("bestIteration", 0),
            "bestValidationScore": model_info.get("bestValidationScore", 0),
            "trainingCurve": {
                "trainMae": training_curve.get("trainMae", []),
                "validMae": training_curve.get("validMae", []),
            },
        },
        "validMetrics": metrics.get("validation", {}),
        "testMetrics": metrics.get("test", {}),
        "baselineComparison": baseline_comparison,
        "modelComparison": model_comparison,
        "baselineImprovement": baseline_improvement,
        "walkForward": metrics.get("walkForward", {}),
        "series": {
            "full": forecast_series.get("full", []),
            "zoom": forecast_series.get("zoom", []),
            "worstPredictions": forecast_series.get("worstPredictions", []),
        },
    }

    # --- errors (reshape for frontend) ---
    errors_obj = {
        "groups": errors.get("groups", {}),
        "heatmap": errors.get("heatmap", {}),
        "series": {
            "full": forecast_series.get("full", []),
            "zoom": forecast_series.get("zoom", []),
            "worstPredictions": forecast_series.get("worstPredictions", []),
        },
    }

    # --- explainability (ensure waterfall field exists) ---
    explainability_obj = dict(explainability)
    if "waterfall" not in explainability_obj:
        explainability_obj["waterfall"] = []

    # --- efficiency (reshape for frontend) ---
    efficiency_obj = {
        "points": [
            {
                "timestamp": p.get("timestamp", ""),
                "actual": p.get("actual_demand_mw", 0),
                "predicted": p.get("expected_demand_mw", 0),
                "residual": p.get("residual_mw", 0),
                "flag": p.get("anomaly_flag", "normal"),
            }
            for p in efficiency.get("points", [])[:800]  # cap for frontend
        ],
        "summary": efficiency.get("summary", {}),
    }

    # --- assemble final payload ---
    return {
        "dataset": dataset_obj,
        "overview": overview_obj,
        "analysis": analysis,
        "statistics": statistics_obj,
        "features": features,
        "forecasting": forecasting_obj,
        "explainability": explainability_obj,
        "errors": errors_obj,
        "efficiency": efficiency_obj,
        "summary": summary,
    }


def _get_analytics(dataset_key: str) -> dict:
    """Load and transform analytics with proper error handling."""
    key = dataset_key.upper()
    try:
        return _build_frontend_payload(key)
    except KeyError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load artifacts for {key}: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Health check with artifact status."""
    available = 0
    if ARTIFACTS_DIR.is_dir():
        available = sum(
            1
            for d in ARTIFACTS_DIR.iterdir()
            if d.is_dir() and (d / "analytics.json").exists()
        )
    return {
        "status": "ok",
        "version": "2.0.0",
        "artifacts_dir": str(ARTIFACTS_DIR),
        "datasets_available": available,
    }


@app.get("/api/datasets")
def list_datasets() -> list[dict]:
    """List all available datasets with metadata."""
    return _discover_datasets()


@app.get("/api/datasets/{dataset_key}/analytics")
def get_analytics(dataset_key: str) -> dict:
    """
    Return the full precomputed analytics payload for a dataset.

    This is the main data source for the frontend dashboard.
    All computation was done offline by the training pipeline.
    """
    try:
        return _get_analytics(dataset_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Artifact load failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/datasets/{dataset_key}/forecast")
def get_forecast(dataset_key: str, horizon: str = "24h") -> dict:
    """
    Return future forecast for a dataset.

    Args:
        horizon: '24h' for next 24 hours, '7d' for next 7 days.
    """
    try:
        data = _load_raw(dataset_key.upper())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if horizon == "24h":
        forecast = data.get("forecast24h", [])
        return {"dataset": dataset_key.upper(), "horizon": "24h", "points": forecast}
    elif horizon == "7d":
        forecast = data.get("forecast7d", [])
        return {"dataset": dataset_key.upper(), "horizon": "7d", "points": forecast}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{horizon}'. Use '24h' or '7d'.",
        )


@app.get("/api/datasets/{dataset_key}/efficiency")
def get_efficiency(dataset_key: str) -> dict:
    """
    Return expected vs actual demand analysis and anomaly detection.

    Shows potential excess-demand signals (not confirmed energy waste).
    """
    try:
        data = _load_raw(dataset_key.upper())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    efficiency = data.get("efficiency", {})
    return {
        "dataset": dataset_key.upper(),
        "points": efficiency.get("points", []),
        "summary": efficiency.get("summary", {}),
    }


@app.get("/api/datasets/{dataset_key}/predict")
def predict(dataset_key: str, date: str) -> dict:
    """
    Predict hourly demand for a specific date.

    Returns predicted and actual values for each hour of the given date.
    Uses the precomputed forecast series from the test split.
    """
    try:
        data = _load_raw(dataset_key.upper())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    series = data.get("forecastSeries", {})
    full_series = series.get("full", [])

    if not full_series:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction series available for {dataset_key.upper()}",
        )

    date_points = [
        p for p in full_series if str(p.get("timestamp", "")).startswith(date)
    ]

    if not date_points:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for date {date} in {dataset_key.upper()}",
        )

    avg_predicted = sum(p.get("predicted", 0) for p in date_points) / len(date_points)
    avg_actual = sum(p.get("actual", 0) for p in date_points) / len(date_points)

    return {
        "dataset": dataset_key.upper(),
        "date": date,
        "split": "test",
        "averagePrediction": round(avg_predicted, 2),
        "averageActual": round(avg_actual, 2),
        "points": date_points,
    }
