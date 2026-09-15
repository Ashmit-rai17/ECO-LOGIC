FROM python:3.11-slim

WORKDIR /app

# Install only inference dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
