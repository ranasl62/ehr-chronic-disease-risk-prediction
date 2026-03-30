# Docker smoke test

Build and run the API container from repository root.

```bash
docker build -f deployment/Dockerfile -t ehr-ai-system .
docker run --rm -p 8000:8000 -v "$(pwd)/model.pkl:/app/model.pkl:ro" ehr-ai-system
```

Then `curl -s http://127.0.0.1:8000/health`.

For API + Streamlit together: `docker compose up --build` (ensure `model.pkl` exists locally for the volume mount).
