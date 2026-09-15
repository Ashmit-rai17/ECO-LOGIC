"""
Offline training configuration.

Centralizes all hyperparameters, paths, and versioning for the ML pipeline.
"""

from pathlib import Path

# ---------- Paths ----------
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

# ---------- Feature Engineering ----------
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [24, 168]
FEATURE_ENGINEERING_VERSION = "1.0.0"

# ---------- Temporal Split ----------
# Train ≤ 2015, Validation = 2016, Test ≥ 2017
TRAIN_END = "2015-12-31"
VALIDATION_START = "2016-01-01"
VALIDATION_END = "2016-12-31"
TEST_START = "2017-01-01"

# ---------- XGBoost ----------
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": 1,
}

# ---------- Walk-Forward ----------
WALK_FORWARD_MIN_TRAIN_YEARS = 2
WALK_FORWARD_TEST_YEARS = 1
WALK_FORWARD_MAX_FOLDS = 6

# ---------- Model Comparison ----------
COMPARE_LINEAR_REGRESSION = True
COMPARE_RANDOM_FOREST = True
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": 1,
}

# ---------- SHAP ----------
SHAP_SAMPLE_SIZE = 300

# ---------- Anomaly Detection ----------
# Context-aware thresholds for anomaly severity classification.
# These are configurable and can be tuned per dataset.
ANOMALY_ZSCORE_ELEVATED = 1.5   # Slightly above expected demand
ANOMALY_ZSCORE_HIGH = 2.5       # Significantly above expected
ANOMALY_ZSCORE_CRITICAL = 3.5   # Extremely unusual demand

# ---------- Weather ----------
WEATHER_CACHE_DIR = BACKEND_DIR / "data" / "weather_cache"

# ---------- Future Forecasting ----------
FORECAST_HORIZON_24H = 24   # hours
FORECAST_HORIZON_7D = 168   # hours

# ---------- Artifact Version ----------
ARTIFACT_VERSION = "1.0.0"
