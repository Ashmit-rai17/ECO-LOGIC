from __future__ import annotations

import gc
from functools import lru_cache

import pandas as pd

from app.artifact_cache import ArtifactCache, dataset_cache_key
from app.config import get_settings
from app.ml.data_loader import DatasetInfo, discover_datasets, load_dataset
from app.ml.evaluation import error_groups
from app.ml.explainability import feature_importance, shap_summary
from app.ml.feature_engineering import build_features, feature_catalog
from app.ml.forecasting import TARGET, compare_models, fit_xgboost, forecast, temporal_split
from app.ml.preprocessing import overview, prepare_time_index
from app.ml.statistical import autocorrelation, correlation_summary, seasonality_insights, stationarity
from app.ml.validation import run_walk_forward
from app.ml.visualization import analysis_payload, error_heatmap, forecast_series


def list_datasets() -> list[DatasetInfo]:
    return discover_datasets(get_settings().data_dir)


def dataset_by_key(key: str) -> DatasetInfo:
    normalized = key.upper()
    for dataset in list_datasets():
        if dataset.key == normalized:
            return dataset
    raise KeyError(f"Unknown dataset: {key}")


@lru_cache(maxsize=12)
def compute_artifacts(key: str) -> dict:
    settings = get_settings()
    info = dataset_by_key(key)
    cache_key = dataset_cache_key(info, settings)
    cache = ArtifactCache(settings.artifacts_dir, settings.cache_enabled)
    cached = cache.load(info.key, cache_key, "training-artifacts")
    if cached is not None:
        return cached

    raw = load_dataset(info)
    df = prepare_time_index(raw)
    del raw
    gc.collect()

    model_df = build_features(df)
    gc.collect()

    # Single split for baseline comparison
    splits = temporal_split(model_df)
    model, model_info = fit_xgboost(splits, settings.xgb_n_estimators, settings.xgb_early_stopping_rounds)
    gc.collect()

    forecast_result = forecast(splits, model)
    gc.collect()

    model_comparison = compare_models(splits, forecast_result["testMetrics"]) if settings.enable_model_comparison else []
    gc.collect()

    dashboard = forecast_result["dashboard"]
    x_valid = splits["valid"].drop(columns=[TARGET])

    # Walk-forward validation
    walk_forward_result = {}
    if settings.enable_walk_forward:
        try:
            walk_forward_result = run_walk_forward(
                model_df,
                settings.xgb_n_estimators,
                settings.xgb_early_stopping_rounds,
                min_train_years=settings.walk_forward_min_train_years,
                test_years=settings.walk_forward_test_years,
            )
        except Exception:
            walk_forward_result = {"folds": [], "aggregated": {}}
        gc.collect()

    artifacts = {
        "info": info,
        "model_df": model_df,
        "splits": splits,
        "model": model,
        "model_info": model_info,
        "forecast_result": forecast_result,
        "model_comparison": model_comparison,
        "dashboard": dashboard,
        "x_valid": x_valid,
        "walk_forward_result": walk_forward_result,
    }
    cache.save(info.key, cache_key, "training-artifacts", artifacts)
    return artifacts


@lru_cache(maxsize=12)
def compute_dataset(key: str) -> dict:
    settings = get_settings()
    info = dataset_by_key(key)
    cache_key = dataset_cache_key(info, settings)
    cache = ArtifactCache(settings.artifacts_dir, settings.cache_enabled)
    cached = cache.load(info.key, cache_key, "dashboard-payload")
    if cached is not None:
        return cached

    artifacts = compute_artifacts(key)
    info = artifacts["info"]
    model_df = artifacts["model_df"]
    splits = artifacts["splits"]
    model = artifacts["model"]
    model_info = artifacts["model_info"]
    forecast_result = artifacts["forecast_result"]
    model_comparison = artifacts["model_comparison"]
    dashboard = artifacts["dashboard"]
    x_valid = artifacts["x_valid"]
    walk_forward_result = artifacts.get("walk_forward_result", {})

    # Reconstruct df for overview/analysis (just the demand column)
    df = model_df[["demand_mw"]].copy()
    gc.collect()

    payload = {
        "dataset": {"key": info.key, "label": info.label, "source": str(info.path)},
        "overview": overview(df, info.key),
        "analysis": analysis_payload(df, settings.max_points),
        "statistics": {
            "stationarity": stationarity(df),
            "correlations": correlation_summary(model_df),
            "seasonalityInsights": seasonality_insights(df),
        },
        "features": {
            "catalog": feature_catalog(),
            "importance": feature_importance(model, forecast_result["xTest"].columns),
        },
        "forecasting": {
            "strategy": {
                "train": {"start": splits["train"].index.min().isoformat(), "end": splits["train"].index.max().isoformat(), "rows": int(len(splits["train"]))},
                "validation": {"start": splits["valid"].index.min().isoformat(), "end": splits["valid"].index.max().isoformat(), "rows": int(len(splits["valid"]))},
                "test": {"start": splits["test"].index.min().isoformat(), "end": splits["test"].index.max().isoformat(), "rows": int(len(splits["test"]))},
            },
            "model": model_info,
            "validMetrics": forecast_result["validMetrics"],
            "testMetrics": forecast_result["testMetrics"],
            "baselineComparison": forecast_result["baselineComparison"],
            "modelComparison": model_comparison,
            "baselineImprovement": forecast_result["baselineImprovement"],
            "walkForward": walk_forward_result,
            "series": forecast_series(dashboard, settings.max_points),
        },
        "explainability": shap_summary(model, x_valid, settings.shap_sample_size) if settings.enable_shap else {"available": False, "message": "SHAP disabled for memory constraints.", "beeswarm": [], "dependence": [], "dependenceFeature": None, "waterfall": []},
        "errors": {
            "groups": error_groups(dashboard),
            "heatmap": error_heatmap(dashboard),
            "series": forecast_series(dashboard, settings.max_points),
        },
        "summary": project_summary(forecast_result["testMetrics"], forecast_result["baselineImprovement"], seasonality_insights(df)),
    }
    del df
    gc.collect()
    cache.save(info.key, cache_key, "dashboard-payload", payload)
    return payload


def predict_for_date(key: str, date: str) -> dict:
    artifacts = compute_artifacts(key.upper())
    model_df = artifacts["model_df"]
    model = artifacts["model"]

    requested = pd.to_datetime(date).date()
    day_rows = model_df.loc[model_df.index.date == requested]
    if day_rows.empty:
        available_start = model_df.index.min().date().isoformat()
        available_end = model_df.index.max().date().isoformat()
        raise ValueError(f"No feature-complete rows for {date}. Choose a date from {available_start} to {available_end}.")

    x_day = day_rows.drop(columns=[TARGET])
    predictions = model.predict(x_day)
    rows = []
    for timestamp, actual, predicted in zip(day_rows.index, day_rows[TARGET], predictions):
        rows.append(
            {
                "timestamp": timestamp.isoformat(),
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(actual - predicted),
                "absoluteError": float(abs(actual - predicted)),
            }
        )

    split = "train"
    if requested >= artifacts["splits"]["test"].index.min().date():
        split = "test"
    elif requested >= artifacts["splits"]["valid"].index.min().date():
        split = "validation"

    return {
        "dataset": key.upper(),
        "date": requested.isoformat(),
        "split": split,
        "points": rows,
        "averagePrediction": float(sum(row["predicted"] for row in rows) / len(rows)),
        "averageActual": float(sum(row["actual"] for row in rows) / len(rows)),
    }


def project_summary(metrics: dict, improvement: float, insights: list[str]) -> dict:
    return {
        "observations": insights,
        "strengths": [
            f"XGBoost improves MAE over the lag-1 baseline by {improvement:.1f}%.",
            "Shifted lag and rolling features capture hourly, daily, and weekly demand memory without target leakage.",
            f"The test WMAPE is {metrics['wmape']:.2f}%, which is easy to communicate to non-technical readers.",
            "Walk-forward validation provides robust multi-fold performance estimates.",
        ],
        "limitations": [
            "The reference workflow uses demand history only; weather, holidays, and market context are not included.",
            "SHAP explanations depend on the local shap package being installed.",
        ],
        "futureImprovements": [
            "Integrate weather data (temperature, humidity, solar irradiance) for exogenous features.",
            "Add HDD/CDD (Heating/Cooling Degree Days) for weather-demand relationships.",
            "Implement Optuna hyperparameter tuning for production-grade model selection.",
        ],
    }
