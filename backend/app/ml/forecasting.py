from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from app.ml.evaluation import baseline_comparison, dashboard_frame, regression_metrics

TARGET = "demand_mw"


def temporal_split(model_df: pd.DataFrame) -> dict:
    return {
        "train": model_df.loc[:"2015-12-31"],
        "valid": model_df.loc["2016-01-01":"2016-12-31"],
        "test": model_df.loc["2017-01-01":],
    }


def fit_xgboost(splits: dict, n_estimators: int, early_stopping_rounds: int) -> tuple[XGBRegressor, dict]:
    train = splits["train"]
    valid = splits["valid"]
    x_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    x_valid = valid.drop(columns=[TARGET])
    y_valid = valid[TARGET]

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], verbose=False)
    return model, {
        "bestIteration": int(getattr(model, "best_iteration", -1)),
        "bestValidationScore": float(getattr(model, "best_score", float("nan"))),
        "trainingCurve": {
            "trainMae": [float(value) for value in model.evals_result()["validation_0"]["mae"]],
            "validMae": [float(value) for value in model.evals_result()["validation_1"]["mae"]],
        },
    }


def forecast(splits: dict, model: XGBRegressor) -> dict:
    valid = splits["valid"]
    test = splits["test"]
    x_valid = valid.drop(columns=[TARGET])
    y_valid = valid[TARGET]
    x_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    valid_predictions = model.predict(x_valid)
    test_predictions = model.predict(x_test)
    metrics = regression_metrics(y_test, test_predictions)
    baseline = baseline_comparison(y_test, x_test, metrics["mae"])
    lag_1_mae = next(row["mae"] for row in baseline if row["model"] == "Lag-1 Baseline")
    improvement = ((lag_1_mae - metrics["mae"]) / lag_1_mae) * 100

    return {
        "validMetrics": regression_metrics(y_valid, valid_predictions),
        "testMetrics": metrics,
        "baselineComparison": baseline,
        "baselineImprovement": float(improvement),
        "dashboard": dashboard_frame(y_test, test_predictions),
        "xTest": x_test,
        "yTest": y_test,
    }


def compare_models(splits: dict, xgboost_metrics: dict) -> list[dict]:
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]
    comparison_train = train.tail(min(35000, len(train)))
    x_train = comparison_train.drop(columns=[TARGET])
    y_train = comparison_train[TARGET]
    x_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    rows = [
        {
            "model": "XGBoost",
            "type": "Gradient boosted trees",
            "status": "trained",
            **xgboost_metrics,
        }
    ]

    candidates = [
        (
            "Linear Regression",
            "Linear regression baseline",
            LinearRegression(),
        ),
        (
            "Random Forest",
            "Bagged decision trees",
            RandomForestRegressor(
                n_estimators=90,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    try:
        from lightgbm import LGBMRegressor

        candidates.append(
            (
                "LightGBM",
                "Gradient boosted trees",
                LGBMRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    except Exception as exc:
        rows.append(
            {
                "model": "LightGBM",
                "type": "Gradient boosted trees",
                "status": f"unavailable: {type(exc).__name__}",
                "mae": None,
                "rmse": None,
                "r2": None,
                "mape": None,
                "wmape": None,
                "meanBias": None,
            }
        )

    rows.append(
        {
            "model": "Logistic Regression",
            "type": "Classification model",
            "status": "not applicable: target is continuous MW demand, not a class label",
            "mae": None,
            "rmse": None,
            "r2": None,
            "mape": None,
            "wmape": None,
            "meanBias": None,
        }
    )

    for model_name, model_type, estimator in candidates:
        try:
            fit_kwargs = {}
            if model_name == "LightGBM":
                fit_kwargs = {"eval_set": [(valid.drop(columns=[TARGET]), valid[TARGET])], "eval_metric": "l1"}
            estimator.fit(x_train, y_train, **fit_kwargs)
            predictions = estimator.predict(x_test)
            rows.append(
                {
                    "model": model_name,
                    "type": model_type,
                    "status": "trained",
                    **regression_metrics(y_test, predictions),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "type": model_type,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "mape": None,
                    "wmape": None,
                    "meanBias": None,
                }
            )

    trained = [row for row in rows if row["status"] == "trained" and row["mae"] is not None]
    if trained:
        best_mae = min(row["mae"] for row in trained)
        for row in rows:
            row["deltaMaeVsBest"] = float(row["mae"] - best_mae) if row["mae"] is not None else None
    return rows
