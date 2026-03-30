# Convenience wrapper — canonical file lives in `deployment/Dockerfile`.
# docker build -t ehr-ai-system .
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt setup.py ./
COPY . .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
