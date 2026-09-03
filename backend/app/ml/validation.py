"""Walk-forward validation for time series forecasting."""

from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from app.ml.evaluation import regression_metrics

TARGET = "demand_mw"


def walk_forward_split(
    model_df: pd.DataFrame,
    min_train_years: int = 2,
    test_years: int = 1,
) -> list[dict]:
    """Generate expanding-window walk-forward splits.

    Returns a list of fold dicts, each containing:
      fold, train_start, train_end, test_start, test_end,
      train_rows, test_rows, train_df, test_df
    """
    years = sorted(model_df.index.year.unique())
    if len(years) < min_train_years + 1:
        return []

    folds = []
    for fold_idx, test_start_year in enumerate(range(
        years[min_train_years], years[-1] + 1, test_years
    )):
        test_end_year = test_start_year + test_years - 1
        train_end_year = test_start_year - 1

        train_df = model_df.loc[:str(train_end_year)]
        test_df = model_df.loc[str(test_start_year):str(test_end_year)]

        if test_df.empty:
            break

        folds.append({
            "fold": fold_idx,
            "train_start": str(train_df.index.year.min()),
            "train_end": str(train_df.index.year.max()),
            "test_start": str(test_df.index.year.min()),
            "test_end": str(test_df.index.year.max()),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_df": train_df,
            "test_df": test_df,
        })

    return folds


def train_fold_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[XGBRegressor, dict]:
    """Train XGBoost on a single fold and return model + metrics.

    Uses 20% of training data as eval set for early stopping.
    """
    # Use last 20% of training data as eval set
    eval_size = max(100, int(len(train_df) * 0.2))
    eval_df = train_df.iloc[-eval_size:]
    train_fit_df = train_df.iloc[:-eval_size]

    if len(train_fit_df) < 50:
        # Not enough data for split, use all training data
        train_fit_df = train_df
        eval_df = None

    x_train = train_fit_df.drop(columns=[TARGET])
    y_train = train_fit_df[TARGET]
    x_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

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
        n_jobs=1,
    )

    eval_set = [(x_train, y_train)]
    if eval_df is not None:
        eval_set.append((eval_df.drop(columns=[TARGET]), eval_df[TARGET]))

    model.fit(x_train, y_train, eval_set=eval_set, verbose=False)

    predictions = model.predict(x_test)
    metrics = regression_metrics(y_test, predictions)

    return model, metrics


def run_walk_forward(
    model_df: pd.DataFrame,
    n_estimators: int,
    early_stopping_rounds: int,
    min_train_years: int = 2,
    test_years: int = 1,
) -> dict:
    """Run full walk-forward validation and return aggregated results.

    Returns:
      folds: list of per-fold results (metrics, periods, best_iteration)
      aggregated: mean and std of each metric across folds
    """
    splits = walk_forward_split(model_df, min_train_years, test_years)
    if not splits:
        return {"folds": [], "aggregated": {}}

    fold_results = []
    all_metrics = {"mae": [], "rmse": [], "r2": [], "mape": [], "wmape": [], "meanBias": []}

    for split in splits:
        model, metrics = train_fold_xgboost(
            split["train_df"],
            split["test_df"],
            n_estimators,
            early_stopping_rounds,
        )

        best_iter = int(getattr(model, "best_iteration", -1))

        fold_result = {
            "fold": split["fold"],
            "trainPeriod": f"{split['train_start']}–{split['train_end']}",
            "testPeriod": f"{split['test_start']}–{split['test_end']}",
            "trainRows": split["train_rows"],
            "testRows": split["test_rows"],
            "bestIteration": best_iter,
            **metrics,
        }
        fold_results.append(fold_result)

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        # Free memory
        del model

    aggregated = {}
    for key, values in all_metrics.items():
        if values:
            aggregated[f"mean_{key}"] = float(pd.Series(values).mean())
            aggregated[f"std_{key}"] = float(pd.Series(values).std()) if len(values) > 1 else 0.0

    return {"folds": fold_results, "aggregated": aggregated}
