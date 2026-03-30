import os
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from inference.predict import load_artifact, risk_level_from_probability
from utils.config import MODEL_PATH
from utils.eval_report import evaluation_aligned_with_manifest, load_evaluation_report_safe

st.set_page_config(
    page_title="Clinical AI Risk Stratification",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("PREDICT_API_URL", "").rstrip("/")
PREDICT_API_KEY = os.environ.get("PREDICT_API_KEY", "").strip()


@st.cache_resource
def _artifact():
    return load_artifact(MODEL_PATH)


@st.cache_data
def _eval_report():
    return load_evaluation_report_safe()


def _synthetic_timeline(values: dict[str, float], cols: list[str]) -> pd.DataFrame:
    """Demo longitudinal view: smooth last 5 visits from current feature snapshot."""
    base = pd.Timestamp.utcnow().normalize()
    times = [base - timedelta(days=30 * i) for i in range(4, -1, -1)]
    rows = []
    for i, t in enumerate(times):
        w = (i + 1) / 5.0
        row = {"visit": i + 1, "charttime": t}
        for c in cols:
            v = float(values.get(c, 0.0))
            noise = 1.0 + 0.03 * (i - 2)
            row[c] = max(0.0, v * noise * w + (0.01 * i if "count" in c else 0))
        rows.append(row)
    return pd.DataFrame(rows)


st.title("Clinical AI Risk Stratification System")
st.caption("Research prototype — not for clinical use. Human oversight required.")

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

try:
    art = _artifact()
except FileNotFoundError:
    st.error(
        f"Missing `{MODEL_PATH.name}`. Train first, e.g. "
        "`python -m training.train --format longitudinal --data data/raw/ehr_data.csv`"
    )
    st.stop()

cols = art["feature_columns"]
fe = art.get("feature_engineering", {})
calibrated = art.get("calibrated", False)
input_stats = art.get("input_stats") or {}

st.sidebar.header("Patient inputs")
st.sidebar.caption("Features must match trained artifact schema.")
st.sidebar.markdown(
    "Data mapping (CMS / claims / MIMIC-style): see `docs/data_sources_and_schema.md` in the repo."
)
if API_URL:
    st.sidebar.caption(f"Using API: `{API_URL}`")
    if PREDICT_API_KEY:
        st.sidebar.caption("`PREDICT_API_KEY` is set (sent as `X-API-Key`).")

use_api_explanation = st.sidebar.checkbox("Include explanation from API (slower)", value=True)

values: dict[str, float] = {}
for c in cols:
    st_meta = input_stats.get(c, {})
    default_f = float(st_meta.get("median", 0.0))
    lo = float(st_meta.get("p05", default_f))
    hi = float(st_meta.get("p95", default_f))
    if lo >= hi:
        lo, hi = default_f - 1.0, default_f + 1.0
    help_txt = (
        f"Training distribution: median {default_f:.4g}; p05–p95 {lo:.4g} … {hi:.4g}"
        if st_meta
        else None
    )
    values[c] = float(
        st.sidebar.number_input(
            f"{c.replace('_', ' ').title()}",
            value=default_f,
            format="%.4f",
            key=f"sb_{c}",
            help=help_txt,
        )
    )

meta = st.sidebar.expander("Model info")
meta.write(f"**Kind:** `{art.get('model_kind', '?')}`")
meta.write(f"**Calibrated:** `{calibrated}`")
meta.write(f"**Artifact:** `{MODEL_PATH}`")
if fe:
    meta.json(fe)
ev = _eval_report()
if ev:
    m = ev.get("metrics") or {}
    aligned = evaluation_aligned_with_manifest(
        art.get("training_manifest"),
        ev.get("meta") or {},
    )
    meta.write(f"**Eval file aligned with artifact:** `{aligned}`")
    meta.caption("`reports/evaluation_report.json` — train after data/model changes.")
    cols_m = ["roc_auc", "pr_auc", "brier", "ece", "f1"]
    parts = [f"{k}={m[k]}" for k in cols_m if k in m]
    if parts:
        meta.write("**Hold-out:** " + " · ".join(parts))
else:
    meta.caption("No `evaluation_report.json` — run `python -m training.train` to generate.")

run = st.sidebar.button("Predict risk", type="primary")

tab_overview, tab_timeline, tab_shap = st.tabs(
    ["Overview", "Patient timeline (demo)", "Explanation (SHAP)"]
)

if run:
    if API_URL:
        try:
            headers = {}
            if PREDICT_API_KEY:
                headers["X-API-Key"] = PREDICT_API_KEY
            payload = {
                "features": values,
                "include_explanation": use_api_explanation,
            }
            r = requests.post(
                f"{API_URL}/v1/predict",
                json=payload,
                headers=headers,
                timeout=60 if use_api_explanation else 30,
            )
            if r.status_code == 401:
                st.error("API returned 401 — set `PREDICT_API_KEY` if the server uses `API_KEY`.")
                st.stop()
            if r.status_code == 400:
                st.error(r.json())
                st.stop()
            r.raise_for_status()
            res = r.json()
            risk = float(res.get("risk_probability", res.get("risk_score", 0.0)))
            level = res.get("risk_level", risk_level_from_probability(risk))
        except Exception as e:
            st.error(f"API error ({API_URL}): {e}")
            st.stop()
    else:
        model = art["model"]
        row = np.array([[values[c] for c in cols]])
        risk = float(model.predict_proba(row)[0, 1])
        level = risk_level_from_probability(risk)

    st.session_state.risk_history.append(risk)
    st.session_state.last_values = values.copy()
    st.session_state.last_risk = risk
    st.session_state.last_level = level

if st.session_state.risk_history:
    risk = st.session_state.last_risk
    level = st.session_state.last_level
    values = st.session_state.last_values

    with tab_overview:
        st.subheader("Risk summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk probability", f"{risk:.3f}")
        c2.metric("Risk band", level)
        c3.metric("Runs in session", len(st.session_state.risk_history))
        if risk > 0.7:
            st.error("High risk — demo band only; requires clinical correlation (not automated action).")
        elif risk > 0.4:
            st.warning("Moderate risk — demo band only.")
        else:
            st.success("Lower risk band — demo only.")

        st.subheader("Session risk trend")
        rh = pd.DataFrame({"run": range(1, len(st.session_state.risk_history) + 1), "risk": st.session_state.risk_history})
        st.line_chart(rh.set_index("run"))

    with tab_timeline:
        st.write("Synthetic multi-visit trend from current sidebar values (illustrative only).")
        tl = _synthetic_timeline(values, cols)
        numeric_cols = [c for c in cols if c in tl.columns]
        if numeric_cols:
            chart_df = tl.set_index("charttime")[numeric_cols]
            st.line_chart(chart_df)
        st.dataframe(tl, use_container_width=True)

    with tab_shap:
        st.write("Local feature contributions (tree / linear base estimator; calibrated wrapper unwrapped when needed).")
        try:
            from explainability.shap_explainer import explain_single_patient

            model = art["model"]
            row_df = pd.DataFrame([values])[cols]
            shap_row = explain_single_patient(model, row_df)
            imp = pd.Series(np.abs(shap_row.ravel()), index=cols).sort_values(ascending=False)
            st.bar_chart(imp)
            st.dataframe(imp.rename("abs_shap").to_frame(), use_container_width=True)
        except Exception as e:
            st.info(f"SHAP panel unavailable in this environment: {e}")
else:
    with tab_overview:
        st.info("Use the sidebar to enter features, then **Predict risk**.")

st.caption(
    "API: `uvicorn api.main:app --reload` · `export PREDICT_API_URL=http://127.0.0.1:8000` · "
    "MIMIC SQL: `sql/feature_queries.sql` · SHAP report: `python scripts/explain_shap.py`"
)
