---
title: EcoLogic ML API
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
license: mit
---

# EcoLogic — Energy Intelligence ML API

Backend ML service for the EcoLogic energy analytics dashboard.

## What it does

- Ingests PJM hourly electricity demand datasets (AEP, COMED, DAYTON, etc.)
- Engineers temporal features (lags, rolling statistics, calendar features)
- Trains XGBoost forecasting models with walk-forward validation
- Generates SHAP explainability and model comparisons
- Serves analytics and prediction results via Gradio API

## API Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/api/list_datasets` | POST | `{}` | List of datasets |
| `/api/get_analytics` | POST | `{"data": ["AEP"]}` | Full analytics JSON |
| `/api/predict` | POST | `{"data": ["AEP", "2018-01-01"]}` | Per-hour predictions |

## Local development

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

## Deployment

This Space uses the **Gradio SDK** with 16 GB RAM (free tier).
The `app.py` file is the entry point — HF Spaces runs it automatically.
