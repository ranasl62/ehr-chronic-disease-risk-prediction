# Angular researcher workbench

Primary UI for this repository (research prototype — not clinical use).

## Dev

```bash
# terminal 1 — API
cd .. && PYTHONPATH=. uvicorn api.main:app --reload --port 8000

# terminal 2 — Angular (proxies /v1 to :8000)
npm start
# http://127.0.0.1:4200
```

## Production build

```bash
# Same-origin (Docker nginx → api, or reverse proxy):
npm ci && npm run build

# Remote API (Vercel / Render UI talking to a separate API host):
API_ENDPOINT=https://ehr-risk-framework.onrender.com npm ci && npm run build
# output: dist/web/browser
```

Docker Compose uses empty `API_ENDPOINT` by default. To bake a remote API into the web image:

```bash
API_ENDPOINT=https://ehr-risk-framework.onrender.com docker compose build web
# or: docker build --build-arg API_ENDPOINT=https://… -t ehr-risk-web ./web
```

On the API host, set `CORS_ORIGINS` to your UI origin when `API_ENDPOINT` is absolute.
