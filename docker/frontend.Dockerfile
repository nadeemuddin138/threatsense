# docker/frontend.Dockerfile
# ThreatSense — Streamlit dashboard image
#
# Build:  docker build -f docker/frontend.Dockerfile -t threatsense-frontend .
# Run:    docker run -p 8501:8501 threatsense-frontend

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Only the frontend and shared src code needed here
COPY frontend/ frontend/
COPY src/      src/

EXPOSE 8501

# Disable Streamlit's browser auto-open and telemetry in containers
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
