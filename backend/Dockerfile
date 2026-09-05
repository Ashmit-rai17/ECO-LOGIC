FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for scikit-learn, xgboost, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and data
COPY . .

# Create artifacts directory for model cache
RUN mkdir -p /app/artifacts

# Cloud Run injects $PORT — default to 8080
ENV PORT=8080
EXPOSE ${PORT}

# Start the API
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 300
