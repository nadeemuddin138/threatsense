# docker/api.Dockerfile
# ThreatSense — FastAPI backend image
#
# Build:  docker build -f docker/api.Dockerfile -t threatsense-api .
# Run:    docker run -p 8000:8000 --env-file .env \
#             -v $(pwd)/models:/app/models \
#             -v $(pwd)/data:/app/data \
#             threatsense-api

FROM python:3.12-slim

# System deps for xgboost / shap native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/  src/
COPY api/  api/

# models/ and data/ are mounted as volumes at runtime — NOT baked in.
# This keeps the image small and lets you update models without rebuilding.

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
