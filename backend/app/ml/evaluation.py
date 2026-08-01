from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: np.ndarray | pd.Series) -> dict:
    predicted_series = pd.Series(predicted, index=actual.index)
    residual = actual - predicted_series
    absolute_error = residual.abs()
    return {
        "mae": float(mean_absolute_error(actual, predicted_series)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted_series))),
        "r2": float(r2_score(actual, predicted_series)),
        "mape": float((absolute_error / actual.replace(0, np.nan).abs()).mean() * 100),
        "wmape": float((absolute_error.sum() / actual.abs().sum()) * 100),
        "meanBias": float(residual.mean()),
    }


def dashboard_frame(actual: pd.Series, predicted: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame({"actual": actual, "predicted": predicted}, index=actual.index)
    output["residual"] = output["actual"] - output["predicted"]
    output["absolute_error"] = output["residual"].abs()
    output["squared_error"] = output["residual"] ** 2
    output["absolute_percentage_error"] = (output["absolute_error"] / output["actual"].replace(0, np.nan).abs()) * 100
    output["hour"] = output.index.hour
    output["dayofweek"] = output.index.dayofweek
    output["month"] = output.index.month
    output["year"] = output.index.year
    output["is_weekend"] = (output["dayofweek"] >= 5).astype(int)
    return output


def baseline_comparison(y_test: pd.Series, x_test: pd.DataFrame, model_mae: float) -> list[dict]:
    rows = []
    for label, column in [
        ("Lag-1 Baseline", "lag_1"),
        ("Lag-24 Baseline", "lag_24"),
        ("Lag-168 Baseline", "lag_168"),
    ]:
        rows.append({"model": label, "mae": float(mean_absolute_error(y_test, x_test[column]))})
    rows.append({"model": "XGBoost", "mae": float(model_mae)})
    return rows


def error_groups(dashboard: pd.DataFrame) -> dict:
    def grouped(column: str) -> list[dict]:
        frame = (
            dashboard.groupby(column)
            .agg(
                mae=("absolute_error", "mean"),
                rmse=("squared_error", lambda values: np.sqrt(values.mean())),
                bias=("residual", "mean"),
            )
            .reset_index()
        )
        return frame.to_dict("records")

    return {
        "byHour": grouped("hour"),
        "byWeekday": grouped("dayofweek"),
        "byMonth": grouped("month"),
    }
