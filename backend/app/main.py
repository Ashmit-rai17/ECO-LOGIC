from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.services.analytics_service import compute_dataset, list_datasets, predict_for_date

settings = get_settings()

app = FastAPI(
    title="Eco Logic Time Series Analytics API",
    version="1.0.0",
    description="FastAPI backend that refactors the notebook workflow into reusable analytics modules.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> list[dict]:
    return [
        {
            "key": dataset.key,
            "label": dataset.label,
            "source": str(dataset.path),
        }
        for dataset in list_datasets()
    ]


@app.get("/api/datasets/{dataset_key}/analytics")
def analytics(dataset_key: str) -> dict:
    try:
        return compute_dataset(dataset_key.upper())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics pipeline failed: {type(exc).__name__}: {exc}") from exc


@app.get("/api/datasets/{dataset_key}/predict")
def predict(dataset_key: str, date: str) -> dict:
    try:
        return predict_for_date(dataset_key.upper(), date)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc
