# Eco Logic Time Series Dashboard

Next.js dashboard with a FastAPI analytics backend for energy demand time-series modeling.

## Project layout

- `frontend/` - Next.js dashboard for Vercel.
- `frontend/src/` - dashboard routes, components, hooks, and API client.
- `backend/` - FastAPI service for Render.
- `backend/app/` - API routes, analytics service, model pipeline, and artifact cache.
- `backend/data/` - local or deployed dataset files. Add `*_hourly.csv` or `*_hourly.csv.zip` files here.
- `backend/artifacts/` - generated model and dashboard caches. This folder is ignored by Git and should be backed by a Render disk in production.
- `render.yaml` - Render blueprint for the backend.
- `frontend/vercel.json` - Vercel build config.

## Local development

Install frontend dependencies:

```bash
npm --prefix frontend install
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Run the API:

```bash
npm run backend
```

Run the dashboard:

```bash
npm run dev
```

The frontend reads the API URL from `NEXT_PUBLIC_API_BASE_URL`; it defaults to `http://127.0.0.1:8000`.

## Deployment

Recommended setup:

- Deploy the Next.js frontend to Vercel.
- Deploy the FastAPI backend to Render.
- In Vercel, use `frontend` as the project root.
- Set `NEXT_PUBLIC_API_BASE_URL` in Vercel to your Render API URL.
- Set `ECOLOGIC_CORS_ORIGINS` in Render to your Vercel app URL.
- Mount a Render disk at `/var/data` so `ECOLOGIC_ARTIFACTS_DIR=/var/data/eco-logic-artifacts` survives restarts.

Backend cache files are keyed by dataset file metadata and model settings. If the dataset or training configuration changes, the backend automatically builds a new artifact cache.
