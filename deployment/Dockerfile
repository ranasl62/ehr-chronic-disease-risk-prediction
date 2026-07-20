# API / training image for Compose (`api`, `prepare`, `train`).
# Build from repository root:
#   docker build -f deployment/Dockerfile -t ehr-risk-api:local .
#
# Published (Docker Hub): ranasl62/ehr-risk-api
# Local stack:      docker compose up --build
# Pull stack:       docker compose -f docker-compose.yml -f docker-compose.publish.yml up

FROM python:3.11-slim

LABEL org.opencontainers.image.title="ehr-risk-api" \
      org.opencontainers.image.description="EHR Risk Framework FastAPI + training image" \
      org.opencontainers.image.source="https://github.com/ranasl62/ehr-chronic-disease-risk-prediction" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Dependency layer first for better cache reuse on CI / Hub builds.
# Runs as root so Compose bind-mounts (data/, reports/, model.pkl) remain writable.
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Non-editable install for publishable images (source still present for scripts / PYTHONPATH).
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
