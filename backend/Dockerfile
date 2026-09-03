FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create data and artifacts directories
RUN mkdir -p /app/data /app/artifacts

# Expose port (HF Spaces uses 7860, Railway/Render use $PORT)
ENV PORT=7860
EXPOSE $PORT

# Start the API
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
