"""
Offline training pipeline for ECO-LOGIC.

Run:
    cd backend && python -m training.train

This script:
1. Discovers and loads all hourly datasets
2. Runs preprocessing and feature engineering
3. Trains XGBoost with temporal split
4. Trains baseline and comparison models
5. Runs walk-forward validation
6. Computes SHAP explainability
7. Generates efficiency intelligence (anomaly detection)
8. Generates future forecasts (24h, 7-day)
9. Saves all artifacts to artifacts/{DATASET_KEY}/

No training happens at API request time.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

# Ensure backend is on sys.path
_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from training.config import (
    ANOMALY_ZSCORE_THRESHOLD,
    ARTIFACT_VERSION,
    ARTIFACTS_DIR,
    COMPARE_LINEAR_REGRESSION,
    COMPARE_RANDOM_FOREST,
    DATA_DIR,
    FEATURE_ENGINEERING_VERSION,
    FORECAST_HORIZON_24H,
    FORECAST_HORIZON_7D,
    RANDOM_FOREST_PARAMS,
    SHAP_SAMPLE_SIZE,
    TEST_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    WALK_FORWARD_MAX_FOLDS,
    WALK_FORWARD_MIN_TRAIN_YEARS,
    WALK_FORWARD_TEST_YEARS,
    XGB_PARAMS,
)
from app.ml.data_loader import DatasetInfo, discover_datasets, load_dataset
from app.ml.evaluation import baseline_comparison, dashboard_frame, regression_metrics
from app.ml.feature_engineering import build_features, feature_catalog
from app.ml.preprocessing import overview, prepare_time_index
from app.ml.statistical import seasonality_insights, stationarity
from app.ml.validation import walk_forward_split

TARGET = "demand_mw"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _temporal_split(model_df: pd.DataFrame) -> dict:
    """Canonical temporal split: train ≤ 2015, valid 2016, test ≥ 2017."""
    return {
        "train": model_df.loc[:TRAIN_END],
        "valid": model_df.loc[VALIDATION_START:VALIDATION_END],
        "test": model_df.loc[TEST_START:],
    }


def _fit_xgboost(train_df, valid_df, test_df, params=None):
    """Train XGBoost and return model + metrics on test set."""
    p = dict(params or XGB_PARAMS)
    early_stop = p.pop("early_stopping_rounds", 50)
    n_jobs = p.pop("n_jobs", 1)
    random_state = p.pop("random_state", 42)

    model = XGBRegressor(
        **p,
        early_stopping_rounds=early_stop,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    x_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    x_valid = valid_df.drop(columns=[TARGET])
    y_valid = valid_df[TARGET]
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        verbose=False,
    )

    valid_pred = model.predict(x_valid)
    test_pred = model.predict(x_test)

    valid_metrics = regression_metrics(y_valid, valid_pred)
    test_metrics = regression_metrics(y_test, test_pred)
    dashboard = dashboard_frame(y_test, test_pred)
    baseline = baseline_comparison(y_test, x_test, test_metrics["mae"])

    return model, {
        "validMetrics": valid_metrics,
        "testMetrics": test_metrics,
        "baselineComparison": baseline,
        "dashboard": dashboard,
        "xTest": x_test,
        "bestIteration": int(getattr(model, "best_iteration", -1)),
        "bestValidationScore": float(getattr(model, "best_score", float("nan"))),
        "trainingCurve": {
            "trainMae": [float(v) for v in model.evals_result()["validation_0"]["mae"]],
            "validMae": [float(v) for v in model.evals_result()["validation_1"]["mae"]],
        },
    }


def _compare_models(train_df, valid_df, test_df, xgboost_test_metrics):
    """Train Linear Regression, Random Forest and return comparison table."""
    # Use last 10k rows for comparison training (memory)
    comp_train = train_df.tail(min(10_000, len(train_df)))
    x_train = comp_train.drop(columns=[TARGET])
    y_train = comp_train[TARGET]
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    rows = [
        {"model": "XGBoost", "type": "Gradient boosted trees", "status": "trained", **xgboost_test_metrics}
    ]

    if COMPARE_LINEAR_REGRESSION:
        try:
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            pred = lr.predict(x_test)
            rows.append({
                "model": "Linear Regression", "type": "Linear baseline",
                "status": "trained", **regression_metrics(y_test, pred),
            })
        except Exception as exc:
            rows.append({
                "model": "Linear Regression", "type": "Linear baseline",
                "status": f"failed: {exc}", **{k: None for k in ["mae", "rmse", "r2", "mape", "wmape", "meanBias"]},
            })

    if COMPARE_RANDOM_FOREST:
        try:
            rf = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
            rf.fit(x_train, y_train)
            pred = rf.predict(x_test)
            rows.append({
                "model": "Random Forest", "type": "Bagged decision trees",
                "status": "trained", **regression_metrics(y_test, pred),
            })
        except Exception as exc:
            rows.append({
                "model": "Random Forest", "type": "Bagged decision trees",
                "status": f"failed: {exc}", **{k: None for k in ["mae", "rmse", "r2", "mape", "wmape", "meanBias"]},
            })

    # Delta vs best
    trained = [r for r in rows if r["status"] == "trained" and r.get("mae") is not None]
    if trained:
        best_mae = min(r["mae"] for r in trained)
        for r in rows:
            r["deltaMaeVsBest"] = float(r["mae"] - best_mae) if r.get("mae") is not None else None

    return rows


def _run_walk_forward(model_df, max_folds=None):
    """Walk-forward validation: expanding window."""
    max_folds = max_folds or WALK_FORWARD_MAX_FOLDS
    splits = walk_forward_split(
        model_df,
        min_train_years=WALK_FORWARD_MIN_TRAIN_YEARS,
        test_years=WALK_FORWARD_TEST_YEARS,
    )
    if not splits:
        return {"folds": [], "aggregated": {}}

    folds = []
    all_metrics = {k: [] for k in ["mae", "rmse", "r2", "mape", "wmape", "meanBias"]}

    for split in splits[:max_folds]:
        eval_size = max(100, int(len(split["train_df"]) * 0.2))
        train_fit = split["train_df"].iloc[:-eval_size]
        eval_df = split["train_df"].iloc[-eval_size:]

        if len(train_fit) < 50:
            train_fit = split["train_df"]
            eval_df = None

        x_train = train_fit.drop(columns=[TARGET])
        y_train = train_fit[TARGET]
        x_test = split["test_df"].drop(columns=[TARGET])
        y_test = split["test_df"][TARGET]

        # Filter keys that will be set explicitly
        _skip = {"early_stopping_rounds", "n_jobs", "random_state"}
        model = XGBRegressor(
            **{k: v for k, v in XGB_PARAMS.items() if k not in _skip},
            early_stopping_rounds=25,
            n_jobs=1,
            random_state=42,
        )

        eval_set = [(x_train, y_train)]
        if eval_df is not None:
            eval_set.append((eval_df.drop(columns=[TARGET]), eval_df[TARGET]))

        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        pred = model.predict(x_test)
        metrics = regression_metrics(y_test, pred)

        folds.append({
            "fold": split["fold"],
            "trainPeriod": f"{split['train_start']}\u2013{split['train_end']}",
            "testPeriod": f"{split['test_start']}\u2013{split['test_end']}",
            "trainRows": split["train_rows"],
            "testRows": split["test_rows"],
            "bestIteration": int(getattr(model, "best_iteration", -1)),
            **metrics,
        })
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

        del model
        gc.collect()

    aggregated = {}
    for k, vals in all_metrics.items():
        if vals:
            aggregated[f"mean_{k}"] = float(pd.Series(vals).mean())
            aggregated[f"std_{k}"] = float(pd.Series(vals).std()) if len(vals) > 1 else 0.0

    return {"folds": folds, "aggregated": aggregated}


def _compute_shap(model, x_valid):
    """Compute SHAP explainability offline."""
    sample = x_valid.tail(min(SHAP_SAMPLE_SIZE, len(x_valid)))
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer(sample)
        shap_df = pd.DataFrame(values.values, columns=sample.columns, index=sample.index)
        mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(15)
        top_feature = str(mean_abs.index[0]) if len(mean_abs) else sample.columns[0]
        dependence = pd.DataFrame({
            "featureValue": sample[top_feature].values,
            "shapValue": shap_df[top_feature].values,
            "timestamp": sample.index.astype(str).values,
        }).sample(min(200, len(sample)), random_state=42)
        waterfall_row = shap_df.iloc[-1].abs().sort_values(ascending=False).head(12).index
        return {
            "available": True,
            "message": "SHAP values computed offline with TreeExplainer.",
            "beeswarm": [{"feature": f, "meanAbsShap": float(v)} for f, v in mean_abs.items()],
            "dependence": dependence.to_dict("records"),
            "dependenceFeature": top_feature,
            "waterfall": [
                {"feature": f, "featureValue": float(sample.iloc[-1][f]), "shapValue": float(shap_df.iloc[-1][f])}
                for f in waterfall_row
            ],
        }
    except Exception as exc:
        return {
            "available": False,
            "message": f"SHAP unavailable: {type(exc).__name__}: {exc}",
            "beeswarm": [], "dependence": [], "dependenceFeature": None, "waterfall": [],
        }


def _generate_future_forecast(model, model_df, forecast_hours):
    """Recursive multi-step forecast using last known lags.

    At each step, we predict demand(t+1), then shift features forward.
    For lag features that are not yet available, we use the prediction.
    For lag features that ARE available (from historical data), we use those.
    """
    last_known = model_df.iloc[-1:].copy()
    predictions = []
    history = model_df[TARGET].values  # all historical values for lags

    for step in range(forecast_hours):
        x_row = last_known.drop(columns=[TARGET]).copy()

        # Predict
        pred = float(model.predict(x_row)[0])
        predictions.append(pred)

        # Build next timestamp
        next_ts = last_known.index + pd.Timedelta(hours=1)
        new_row = last_known.copy()
        new_row.index = next_ts
        new_row[TARGET] = pred

        # Update datetime features for the new row
        new_row["hour"] = next_ts.hour
        new_row["dayofweek"] = next_ts.dayofweek
        new_row["dayofyear"] = next_ts.dayofyear
        new_row["month"] = next_ts.month
        new_row["quarter"] = next_ts.quarter
        new_row["year"] = next_ts.year
        new_row["is_weekend"] = int(next_ts.dayofweek >= 5)
        new_row["hour_sin"] = np.sin(2 * np.pi * next_ts.hour / 24)
        new_row["hour_cos"] = np.cos(2 * np.pi * next_ts.hour / 24)
        new_row["month_sin"] = np.sin(2 * np.pi * next_ts.month / 12)
        new_row["month_cos"] = np.cos(2 * np.pi * next_ts.month / 12)
        new_row["dayofweek_sin"] = np.sin(2 * np.pi * next_ts.dayofweek / 7)
        new_row["dayofweek_cos"] = np.cos(2 * np.pi * next_ts.dayofweek / 7)

        # Update lag features
        all_with_pred = np.append(history, predictions)
        for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
            idx = len(all_with_pred) - lag - 1
            if idx >= 0:
                new_row[f"lag_{lag}"] = all_with_pred[idx]

        # Update rolling features (use prediction in window)
        window_24 = all_with_pred[-25:]  # last 24+1 for shifted
        window_168 = all_with_pred[-169:]
        new_row["rolling_mean_24"] = float(np.mean(window_24[:-1]))
        new_row["rolling_std_24"] = float(np.std(window_24[:-1])) if len(window_24) > 1 else 0
        new_row["rolling_min_24"] = float(np.min(window_24[:-1]))
        new_row["rolling_max_24"] = float(np.max(window_24[:-1]))
        new_row["rolling_mean_168"] = float(np.mean(window_168[:-1])) if len(window_168) > 1 else float(np.mean(window_24[:-1]))
        new_row["rolling_std_168"] = float(np.std(window_168[:-1])) if len(window_168) > 2 else 0

        last_known = new_row

    last_ts = model_df.index[-1]
    forecast_timestamps = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=forecast_hours, freq="h")

    return [
        {"timestamp": ts.isoformat(), "predicted_demand_mw": float(pred)}
        for ts, pred in zip(forecast_timestamps, predictions)
    ]


def _generate_efficiency_intelligence(dashboard_df, model_df):
    """Expected vs actual demand + anomaly detection.

    Uses the model's test predictions as 'expected demand'.
    Residuals > threshold z-scores are flagged as anomalies.
    """
    actual = dashboard_df["actual"].values
    predicted = dashboard_df["predicted"].values
    residuals = actual - predicted

    mean_resid = float(np.mean(residuals))
    std_resid = float(np.std(residuals))

    anomaly_flags = []
    for r in residuals:
        z = (r - mean_resid) / std_resid if std_resid > 0 else 0
        if z > ANOMALY_ZSCORE_THRESHOLD:
            flag = "high_anomaly"
        elif z > 1.5:
            flag = "elevated_demand"
        else:
            flag = "normal"
        anomaly_flags.append(flag)

    result = []
    for ts, act, exp, resid, flag in zip(
        dashboard_df.index, actual, predicted, residuals, anomaly_flags
    ):
        result.append({
            "timestamp": ts.isoformat(),
            "actual_demand_mw": float(act),
            "expected_demand_mw": float(exp),
            "residual_mw": float(resid),
            "anomaly_flag": flag,
        })

    summary = {
        "total_observations": len(result),
        "normal": sum(1 for r in result if r["anomaly_flag"] == "normal"),
        "elevated_demand": sum(1 for r in result if r["anomaly_flag"] == "elevated_demand"),
        "high_anomaly": sum(1 for r in result if r["anomaly_flag"] == "high_anomaly"),
        "mean_residual_mw": mean_resid,
        "std_residual_mw": std_resid,
    }

    return {"points": result, "summary": summary}


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train_dataset(info: DatasetInfo) -> dict:
    """Full training pipeline for a single dataset. Returns artifact dict."""
    print(f"\n{'='*60}")
    print(f"  Training: {info.key}")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. Load and preprocess
    raw = load_dataset(info)
    df = prepare_time_index(raw)
    del raw; gc.collect()

    # 2. Overview
    ov = overview(df, info.key)
    print(f"  Loaded {ov['observations']} observations from {ov['timeRange']['start']} to {ov['timeRange']['end']}")

    # 3. Feature engineering
    model_df = build_features(df)
    gc.collect()
    print(f"  Features: {len(model_df.columns)} columns after engineering")

    # 4. Temporal split
    splits = _temporal_split(model_df)
    print(f"  Split — train: {len(splits['train'])}, valid: {len(splits['valid'])}, test: {len(splits['test'])}")

    # Handle short datasets (e.g. NI ends 2011 — no valid/test)
    if len(splits["valid"]) == 0 or len(splits["test"]) == 0:
        print(f"  WARNING: Insufficient data for standard split. Using last 20% as test.")
        cutoff = int(len(model_df) * 0.8)
        splits = {
            "train": model_df.iloc[:cutoff],
            "valid": model_df.iloc[cutoff:cutoff + int(len(model_df) * 0.1)],
            "test": model_df.iloc[cutoff + int(len(model_df) * 0.1):],
        }
        if len(splits["valid"]) == 0:
            splits["valid"] = splits["train"].tail(int(len(splits["train"]) * 0.2))
            splits["train"] = splits["train"].iloc[:-len(splits["valid"])]
        print(f"  Fallback split — train: {len(splits['train'])}, valid: {len(splits['valid'])}, test: {len(splits['test'])}")

    # 5. XGBoost training
    model, train_result = _fit_xgboost(
        splits["train"], splits["valid"], splits["test"]
    )
    test_metrics = train_result["testMetrics"]
    print(f"  XGBoost test MAE: {test_metrics['mae']:.1f} MW, R²: {test_metrics['r2']:.4f}")

    # 6. Model comparison
    comparison = _compare_models(
        splits["train"], splits["valid"], splits["test"], test_metrics
    )
    print(f"  Compared {len(comparison)} models")

    # 7. Walk-forward
    wf = _run_walk_forward(model_df)
    if wf["folds"]:
        print(f"  Walk-forward: {len(wf['folds'])} folds, mean MAE: {wf['aggregated'].get('mean_mae', 'N/A'):.1f}")
    gc.collect()

    # 8. SHAP
    x_valid = splits["valid"].drop(columns=[TARGET])
    shap_result = _compute_shap(model, x_valid)
    print(f"  SHAP: {'available' if shap_result['available'] else 'unavailable'}")
    gc.collect()

    # 9. Future forecasts
    forecast_24h = _generate_future_forecast(model, model_df, FORECAST_HORIZON_24H)
    forecast_7d = _generate_future_forecast(model, model_df, FORECAST_HORIZON_7D)
    print(f"  Forecasts: 24h ({len(forecast_24h)} points), 7d ({len(forecast_7d)} points)")

    # 10. Efficiency intelligence
    efficiency = _generate_efficiency_intelligence(train_result["dashboard"], model_df)
    print(f"  Efficiency: {efficiency['summary']['high_anomaly']} anomalies, {efficiency['summary']['elevated_demand']} elevated")

    # 11. Statistics
    stats = stationarity(df)
    seasonality = seasonality_insights(df)

    # 12. Build analytics payload (what the API serves)
    from app.ml.visualization import analysis_payload, error_heatmap, forecast_series, sample_timeseries
    max_points = 800

    # Aggregate artifacts
    elapsed = time.time() - t0
    artifacts = {
        # Metadata
        "metadata": {
            "dataset_key": info.key,
            "dataset_label": info.label,
            "artifact_version": ARTIFACT_VERSION,
            "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
            "training_date": pd.Timestamp.now().isoformat(),
            "training_duration_seconds": round(elapsed, 1),
            "features": list(model_df.columns),
            "target": TARGET,
            "model_type": "XGBoost",
            "model_hyperparameters": {k: v for k, v in XGB_PARAMS.items()},
            "train_period": {"start": str(splits["train"].index.min()), "end": str(splits["train"].index.max()), "rows": len(splits["train"])},
            "validation_period": {"start": str(splits["valid"].index.min()), "end": str(splits["valid"].index.max()), "rows": len(splits["valid"])},
            "test_period": {"start": str(splits["test"].index.min()), "end": str(splits["test"].index.max()), "rows": len(splits["test"])},
            "weather_features": False,
        },

        # Metrics
        "metrics": {
            "train": {"rows": len(splits["train"])},
            "validation": train_result["validMetrics"],
            "test": test_metrics,
            "walkForward": wf,
        },

        # Model comparison
        "modelComparison": comparison,

        # Baselines
        "baselineComparison": train_result["baselineComparison"],
        "baselineImprovement": float(
            ((train_result["baselineComparison"][0]["mae"] - test_metrics["mae"])
             / train_result["baselineComparison"][0]["mae"]) * 100
        ),

        # Training info
        "modelInfo": {
            "bestIteration": train_result["bestIteration"],
            "bestValidationScore": train_result["bestValidationScore"],
            "trainingCurve": train_result["trainingCurve"],
        },

        # Explainability
        "explainability": shap_result,

        # Efficiency intelligence
        "efficiency": efficiency,

        # Future forecasts
        "forecast24h": forecast_24h,
        "forecast7d": forecast_7d,

        # Overview & analysis (compact, precomputed)
        "overview": ov,
        "analysis": analysis_payload(df[["demand_mw"]], max_points),

        # Statistics
        "statistics": {
            "stationarity": stats,
            "correlations": [{"feature": c["feature"], "correlation": c["correlation"]}
                             for c in _correlation_summary(model_df)],
            "seasonalityInsights": seasonality,
        },

        # Features
        "features": {
            "catalog": feature_catalog(),
            "importance": _feature_importance_list(model, model_df.drop(columns=[TARGET]).columns),
        },

        # Error analysis
        "errors": {
            "groups": _error_groups(train_result["dashboard"]),
            "heatmap": error_heatmap(train_result["dashboard"]),
            "series": forecast_series(train_result["dashboard"], max_points),
        },

        # Forecast series for plotting
        "forecastSeries": forecast_series(train_result["dashboard"], max_points),

        # Summary
        "summary": _project_summary(test_metrics, train_result, seasonality),
    }

    # Save
    out_dir = ARTIFACTS_DIR / info.key
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "analytics.json"
    with open(artifact_path, "w") as f:
        json.dump(artifacts, f, default=str, indent=2)
    print(f"  Saved: {artifact_path} ({artifact_path.stat().st_size / 1024:.0f} KB)")
    print(f"  Time: {elapsed:.1f}s")

    # Cleanup
    del model, splits, model_df, train_result
    gc.collect()

    return artifacts


def _correlation_summary(model_df):
    corr = model_df.corr(numeric_only=True)["demand_mw"].drop("demand_mw").sort_values(key=lambda v: v.abs(), ascending=False)
    return [{"feature": k, "correlation": float(v)} for k, v in corr.head(15).items()]


def _feature_importance_list(model, columns):
    frame = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    frame = frame.sort_values("importance", ascending=False)
    return [{"feature": row.feature, "importance": float(row.importance)} for row in frame.itertuples()]


def _error_groups(dashboard):
    def grouped(col):
        return (
            dashboard.groupby(col)
            .agg(mae=("absolute_error", "mean"),
                 rmse=("squared_error", lambda v: float(np.sqrt(v.mean()))),
                 bias=("residual", "mean"))
            .reset_index()
            .to_dict("records")
        )
    return {"byHour": grouped("hour"), "byWeekday": grouped("dayofweek"), "byMonth": grouped("month")}


def _project_summary(test_metrics, train_result, seasonality):
    improvement = train_result["baselineComparison"][0]["mae"] - test_metrics["mae"]
    pct = (improvement / train_result["baselineComparison"][0]["mae"]) * 100
    return {
        "observations": seasonality,
        "strengths": [
            f"XGBoost improves MAE over the lag-1 baseline by {pct:.1f}%.",
            "Shifted lag and rolling features capture hourly, daily, and weekly demand memory without target leakage.",
            f"The test WMAPE is {test_metrics['wmape']:.2f}%, which is easy to communicate to non-technical readers.",
            "Walk-forward validation provides robust multi-fold performance estimates.",
        ],
        "limitations": [
            "The current model uses demand history only; weather, holidays, and market context are not included.",
            "Recursive multi-step forecasting accumulates prediction error over longer horizons.",
            "Potential inefficiency signals are based on statistical residuals, not physical energy audit data.",
        ],
        "futureImprovements": [
            "Integrate weather data (temperature, humidity, solar irradiance) for exogenous features.",
            "Add HDD/CDD (Heating/Cooling Degree Hours) for weather-demand relationships.",
            "Compare demand-only vs weather-enhanced models with proper A/B evaluation.",
            "Implement Optuna hyperparameter tuning for production-grade model selection.",
        ],
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    datasets = discover_datasets(DATA_DIR)
    if not datasets:
        print("ERROR: No datasets found in", DATA_DIR)
        sys.exit(1)

    print(f"Found {len(datasets)} datasets: {[d.key for d in datasets]}")

    results = {}
    for info in datasets:
        try:
            results[info.key] = train_dataset(info)
        except Exception as exc:
            print(f"  FAILED: {info.key} — {type(exc).__name__}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE: {len(results)}/{len(datasets)} succeeded")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
