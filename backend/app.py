"""
EcoLogic ML API — Gradio wrapper for Hugging Face Spaces deployment.

This module wraps the existing FastAPI/ML pipeline into Gradio functions
that HF Spaces can serve with the free Gradio SDK (16 GB RAM).
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the backend package is importable when HF Spaces runs app.py
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradio as gr  # noqa: E402

from app.services.analytics_service import (  # noqa: E402
    compute_dataset,
    list_datasets,
    predict_for_date,
)

# ---------------------------------------------------------------------------
# Gradio-wrapped functions
# ---------------------------------------------------------------------------

def api_list_datasets() -> str:
    """Return the list of available datasets as JSON."""
    datasets = list_datasets()
    return json.dumps(
        [{"key": d.key, "label": d.label, "source": str(d.path)} for d in datasets]
    )


def api_get_analytics(dataset_name: str) -> str:
    """Run the full analytics pipeline for *dataset_name* and return JSON."""
    result = compute_dataset(dataset_name.strip().upper())
    return json.dumps(result)


def api_predict(dataset_name: str, date: str) -> str:
    """Return per-hour predictions for the given date and dataset."""
    result = predict_for_date(dataset_name.strip().upper(), date.strip())
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="EcoLogic ML API",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# ⚡ EcoLogic — Energy Intelligence API\n"
        "Select a dataset, then request analytics or a single-day prediction.\n"
        "Each function is also exposed as an HTTP API endpoint."
    )

    with gr.Tab("📊 Analytics"):
        ds_dropdown = gr.Dropdown(
            label="Dataset",
            choices=[d.key for d in list_datasets()],
            value=None,
            interactive=True,
        )
        load_btn = gr.Button("Load Analytics", variant="primary")
        analytics_output = gr.JSON(label="Analytics Result")
        load_btn.click(
            fn=api_get_analytics,
            inputs=ds_dropdown,
            outputs=analytics_output,
            api_name="get_analytics",
        )

    with gr.Tab("🔮 Predict"):
        pred_ds = gr.Dropdown(
            label="Dataset",
            choices=[d.key for d in list_datasets()],
            value=None,
            interactive=True,
        )
        pred_date = gr.Textbox(label="Date (YYYY-MM-DD)", value="2018-01-01")
        pred_btn = gr.Button("Predict", variant="primary")
        pred_output = gr.JSON(label="Prediction Result")
        pred_btn.click(
            fn=api_predict,
            inputs=[pred_ds, pred_date],
            outputs=pred_output,
            api_name="predict",
        )

    # Hidden component to register list_datasets as a callable API endpoint
    _hidden_trigger = gr.Button(visible=False)
    _hidden_output = gr.Text(visible=False)
    _hidden_trigger.click(
        fn=api_list_datasets,
        inputs=[],
        outputs=_hidden_output,
        api_name="list_datasets",
    )

    gr.Markdown(
        "---\n"
        "**HTTP API** — call these `POST` endpoints directly from your frontend:\n"
        "```\n"
        "POST /api/list_datasets       → [{key, label, source}]\n"
        "POST /api/get_analytics       → full analytics JSON\n"
        "POST /api/predict             → per-hour predictions\n"
        "```\n"
        "Request body format: `{\"data\": [arg1, arg2, ...]}`\n"
    )

# HF Spaces Gradio SDK expects a top-level `demo` variable.
# The three API functions are registered via .click() with explicit
# api_name parameters, making them callable at:
#   POST /api/list_datasets
#   POST /api/get_analytics
#   POST /api/predict
