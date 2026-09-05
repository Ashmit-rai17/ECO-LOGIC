"""
EcoLogic ML API — Gradio wrapper for Hugging Face Spaces deployment.

All heavy imports are deferred so the Gradio UI renders immediately.
The ML pipeline loads on first request.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Monkey-patch: gradio==5.0.0 (forced by HF Spaces) imports HfFolder from
# huggingface_hub, but it was removed in huggingface_hub>=1.0.
# ---------------------------------------------------------------------------
import huggingface_hub as _hf_hub

if not hasattr(_hf_hub, "HfFolder"):
    class _HfFolder:
        _token: str | None = None

        @classmethod
        def get_token(cls) -> str | None:
            return cls._token

        @classmethod
        def set_token(cls, token: str, complete: bool = False) -> None:
            cls._token = token

    _hf_hub.HfFolder = _HfFolder  # type: ignore[attr-defined]
# ---------------------------------------------------------------------------

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradio as gr  # noqa: E402


# ---------------------------------------------------------------------------
# Lazy helpers — heavy imports happen only when a function is called
# ---------------------------------------------------------------------------

def _get_dataset_choices() -> list[str]:
    """Discover datasets on first call (heavy import)."""
    from app.services.analytics_service import list_datasets  # noqa: delayed
    return [d.key for d in list_datasets()]


def api_list_datasets() -> str:
    from app.services.analytics_service import list_datasets
    datasets = list_datasets()
    return json.dumps(
        [{"key": d.key, "label": d.label, "source": str(d.path)} for d in datasets]
    )


def api_get_analytics(dataset_name: str) -> str:
    from app.services.analytics_service import compute_dataset
    result = compute_dataset(dataset_name.strip().upper())
    return json.dumps(result)


def api_predict(dataset_name: str, date: str) -> str:
    from app.services.analytics_service import predict_for_date
    result = predict_for_date(dataset_name.strip().upper(), date.strip())
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Gradio Blocks UI — no heavy imports at module level
# ---------------------------------------------------------------------------

with gr.Blocks(title="EcoLogic ML API", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# ⚡ EcoLogic — Energy Intelligence API\n"
        "Select a dataset, then request analytics or a single-day prediction.\n"
        "Each function is also exposed as an HTTP API endpoint."
    )

    with gr.Tab("📊 Analytics"):
        ds_dropdown = gr.Dropdown(
            label="Dataset",
            choices=[],  # populated lazily
            value=None,
            interactive=True,
        )
        load_btn = gr.Button("Load Analytics", variant="primary")
        analytics_output = gr.JSON(label="Analytics Result")

        def _load_datasets():
            return gr.update(choices=_get_dataset_choices())

        ds_dropdown.change(fn=_load_datasets, inputs=[], outputs=[ds_dropdown])
        load_btn.click(
            fn=api_get_analytics,
            inputs=ds_dropdown,
            outputs=analytics_output,
            api_name="get_analytics",
        )

    with gr.Tab("🔮 Predict"):
        pred_ds = gr.Dropdown(
            label="Dataset",
            choices=[],
            value=None,
            interactive=True,
        )
        pred_date = gr.Textbox(label="Date (YYYY-MM-DD)", value="2018-01-01")
        pred_btn = gr.Button("Predict", variant="primary")
        pred_output = gr.JSON(label="Prediction Result")

        def _load_datasets_pred():
            return gr.update(choices=_get_dataset_choices())

        pred_ds.change(fn=_load_datasets_pred, inputs=[], outputs=[pred_ds])
        pred_btn.click(
            fn=api_predict,
            inputs=[pred_ds, pred_date],
            outputs=pred_output,
            api_name="predict",
        )

    # Register list_datasets as a callable API endpoint
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
        "**HTTP API** — POST endpoints:\n"
        "```\n"
        "POST /api/list_datasets       → [{key, label, source}]\n"
        "POST /api/get_analytics       → full analytics JSON\n"
        "POST /api/predict             → per-hour predictions\n"
        "```\n"
        'Body format: `{"data": [arg1, arg2, ...]}`\n'
    )
