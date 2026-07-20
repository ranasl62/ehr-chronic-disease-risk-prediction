# How it helps — and how to give feedback

**Maintainer:** Md Rana Hossain  
**Email:** [support@larucare.com](mailto:support@larucare.com)  
**LinkedIn:** [mdranahossain](https://www.linkedin.com/in/mdranahossain/)

This page explains **who this framework is for**, **what problem it solves**, and **how to send feedback** so the project stays useful.

---

## Who it helps

| You are… | You get… |
|----------|----------|
| Health-informatics / ML student or researcher | A leakage-aware train → audit → explain → serve loop without starting from a blank notebook |
| Lab building chronic-risk prototypes | Task YAML, Docker one-command demo, Angular workbench, downloadable results ZIP |
| Engineer integrating EHR-style CSV | Clear longitudinal schema, map-preview import, dataset health gates |
| Reviewer of ML methodology | Calibration, leakage audit, SHAP, and explicit [`LIMITATIONS.md`](../LIMITATIONS.md) |

**Not for:** unsupervised clinical diagnosis, FDA/device claims, or replacing a hospital EHR.

---

## How it helps (concrete)

1. **Fewer silent temporal bugs** — index/horizon truncation + automated leakage audit catch common “future leaked into past” mistakes.
2. **Faster first success** — `docker compose up` → UI on :8080 → demo train → predict in minutes.
3. **Comparable experiments** — multi-model compare, named runs, promote-to-active, manifests with data hashes.
4. **Explainable outputs** — SHAP + schema-driven predict forms for stakeholder demos (research context).
5. **Honest defaults** — tiny-cohort warnings, health blockers, disclaimers; you are not sold a clinical product.

Typical workflow:

```text
Datasets (import + health) → Train / Compare → Analytics & Results → Predict
```

Details: [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`docs/researcher_quickstart.md`](researcher_quickstart.md) · **website** [`docs/`](./).

---

## If you use it — please tell us

Feedback makes the framework better for the next user. Useful reports include:

- What you tried (Docker / CLI / Angular)
- Dataset type (demo / synthetic / BYO — **no PHI in email or GitHub issues**)
- What worked and what blocked you
- Screenshots of UI issues (redact any real identifiers)
- Feature requests that fit the niche (leakage, calibration, researcher UX) — see [`LIMITATIONS.md`](../LIMITATIONS.md) for what we will not claim

### Channels

| Channel | Use for |
|---------|---------|
| **Email** [support@larucare.com](mailto:support@larucare.com) | Questions, collaboration, install help |
| **GitHub Issues** | Bugs and public feature requests (templates under `.github/ISSUE_TEMPLATE/`) |
| **Pull requests** | Fixes and small docs improvements ([`CONTRIBUTING.md`](../CONTRIBUTING.md)) |

Suggested email subject: `[ehr-risk-framework] feedback — <short topic>`

### Cite the software

If the framework helped your project or thesis, please cite [`CITATION.cff`](../CITATION.cff). That visibility helps others find a leakage-aware baseline.

---

## Thank you

Every report of a confusing error message, a missing map alias, or a clearer docs paragraph improves the toolkit for the community. Contact either email above anytime.
