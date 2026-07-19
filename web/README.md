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
npm ci && npm run build
# output: dist/web/browser
```

Docker: built by root `docker compose` service `web` on port **8080**.
