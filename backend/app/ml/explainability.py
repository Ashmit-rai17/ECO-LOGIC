from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def feature_importance(model: XGBRegressor, columns: pd.Index) -> list[dict]:
    frame = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    frame = frame.sort_values("importance", ascending=False)
    return [{"feature": row.feature, "importance": float(row.importance)} for row in frame.itertuples()]


def shap_summary(model: XGBRegressor, x_valid: pd.DataFrame, sample_size: int) -> dict:
    sample = x_valid.tail(min(sample_size, len(x_valid)))
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer(sample)
        shap_frame = pd.DataFrame(values.values, columns=sample.columns, index=sample.index)
        mean_abs = shap_frame.abs().mean().sort_values(ascending=False).head(15)
        top_feature = str(mean_abs.index[0]) if len(mean_abs) else sample.columns[0]
        dependence = pd.DataFrame(
            {
                "featureValue": sample[top_feature],
                "shapValue": shap_frame[top_feature],
                "timestamp": sample.index.astype(str),
            }
        ).sample(min(200, len(sample)), random_state=42)
        waterfall_row = shap_frame.iloc[-1].abs().sort_values(ascending=False).head(12).index
        return {
            "available": True,
            "message": "SHAP values computed with TreeExplainer on the validation sample.",
            "beeswarm": [{"feature": feature, "meanAbsShap": float(value)} for feature, value in mean_abs.items()],
            "dependence": dependence.to_dict("records"),
            "dependenceFeature": top_feature,
            "waterfall": [
                {
                    "feature": feature,
                    "featureValue": float(sample.iloc[-1][feature]),
                    "shapValue": float(shap_frame.iloc[-1][feature]),
                }
                for feature in waterfall_row
            ],
        }
    except Exception as exc:
        return {
            "available": False,
            "message": f"SHAP is not available in this environment ({type(exc).__name__}: {exc}). Feature importance is still returned.",
            "beeswarm": [],
            "dependence": [],
            "dependenceFeature": None,
            "waterfall": [],
        }
