# Convenience wrapper — canonical API image is `deployment/Dockerfile`.
# Prefer Compose (builds both API + Angular):
#   docker compose up --build
#
# Pull published images (no local build):
#   docker compose -f docker-compose.yml -f docker-compose.publish.yml pull && up
#
# API-only image:
#   docker build -f deployment/Dockerfile -t ehr-risk-api:local .
#   # or: docker build -t ehr-risk-api:local .

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
# Runtime (override with -e / Compose / platform env): API_KEY, CORS_ORIGINS, MODEL_PATH, …

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
