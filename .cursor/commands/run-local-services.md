# Run API and dashboard locally

Two terminals from repo root (after `model.pkl` exists).

**API**

```bash
uvicorn api.main:app --reload
```

**Dashboard (local model)**

```bash
streamlit run dashboard/app.py
```

**Dashboard via API**

```bash
export PREDICT_API_URL=http://127.0.0.1:8000
streamlit run dashboard/app.py
```

Smoke-test: `GET http://127.0.0.1:8000/health`, `GET /v1/model/schema`, `POST /v1/predict` with a valid feature dict.
