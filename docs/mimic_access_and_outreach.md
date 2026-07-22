# MIMIC-IV access, outreach, and test plan

Action pack for the deferred **H1 credentialed evaluation** in the paper.  
**Never commit MIMIC row data.** Public repo may only hold SHA locks and aggregate metrics under governance.

Related: [`mimic_lock_checklist.md`](mimic_lock_checklist.md) · [`mimic_extract_splits_runbook.md`](mimic_extract_splits_runbook.md) · [`mimic_results_lock.md`](mimic_results_lock.md) · [`external_validation.md`](external_validation.md)

---

## 1. Start with MIMIC-IV (PhysioNet)

Fastest standard path to high-quality real EHR for this stack:

1. Create / sign in at [PhysioNet](https://physionet.org/).
2. Complete required **CITI** training via [PhysioNet training settings](https://physionet.org/settings/training/).
3. Request **MIMIC-IV** access and sign the **Data Use Agreement (DUA)**.
4. Wait for credentialing approval (often days–weeks).
5. Download / load MIMIC-IV locally (Postgres or institution-approved warehouse).
6. Confirm institutional rules (IRB / HIPAA / restricted-data policy) **outside** this repo.

Then follow [`mimic_lock_checklist.md`](mimic_lock_checklist.md) for extract → validate → lock.

| Do | Do not |
|----|--------|
| Keep extracts under `data/processed/` (gitignored) | Push CSVs or PHI to GitHub |
| Publish only `cohort_lock.json` (SHA) + approved aggregates | Invent MIMIC AUCs before a locked run |
| Use the workbench / CLI with the same index–horizon contract as the paper | Change the estimand mid-study without documenting it |

---

## 2. Email template (researcher / collaborator outreach)

Adapt names, institution, and ask. Keep it short; attach or link the one-page test plan below.

```text
Subject: Collaboration request — leakage-aware EHR risk workbench + MIMIC-IV evaluation

Dear Dr. [Last name],

I am Md Rana Hossain (Maharishi International University). I maintain the open-source
EHR Risk Framework (DOI 10.5281/zenodo.21448693; https://github.com/ranasl62/ehr-risk-framework),
a research workbench for index/horizon risk modeling with fail-closed leakage audits,
calibration (Brier/ECE), and SHAP/fairness artifacts. The public paper verifies the
software on synthetic data only; credentialed evaluation is the next step.

I am completing PhysioNet CITI training and the MIMIC-IV DUA. I am writing to ask whether
you would be open to a brief conversation about [one of]:
  (a) informal advice on MIMIC cohort design for a 365-day horizon task, or
  (b) a possible collaboration on a governed MIMIC-IV evaluation under your IRB/DUA rules.

I am not requesting shared PHI by email. Any data stay local under PhysioNet DUA and
institutional policy. Metrics that leave the credentialed environment would be aggregates
or SHA locks only.

A one-page test plan is below / attached. Happy to adapt to your preferred task
(e.g., diabetes risk, 30-day readmission).

Thank you for your time.
Sincerely,
Md Rana Hossain
ORCID: https://orcid.org/0009-0005-5996-719X
Email: mdrana.hossain@miu.edu · support@larucare.com
```

**Follow-up:** polite bump every **2–3 weeks** if no reply. One short paragraph, same subject line + “Following up”.

---

## 3. Test plan (shareable, one page)

**Goal.** Run one governed MIMIC-IV cohort through the EHR Risk Framework and lock reproducible integrity + illustrative metrics—without claiming clinical deployment readiness.

**Estimand (freeze before extract).**
- Population: adults in MIMIC-IV with documented index time (define inclusion/exclusion locally).
- Index: `index_time` (e.g., discharge or first qualifying encounter—pick one and keep it).
- Horizon: **H = 365 days** (or **30 days** for readmission); features **t ≤ t_index**; labels only in **(t_index, t_index+H]**.
- Outcome: binary `label` (define ICD / event rule in a local protocol note).

**Pipeline (software).**
1. Extract → longitudinal CSV (`patient_id`, `timestamp`, `index_time`, labs/vitals as available, `label`).
2. `scripts/validate_training_data.py` + health gate.
3. Train matrix: logistic regression / random forest / XGBoost × windows `{180}` and `{7,30,180}` × isotonic on/off (subset OK if compute-limited).
4. Required: patient-disjoint (or temporal) split; leakage audit must **pass** on integrity path.
5. Report ROC-AUC, PR-AUC, Brier, ECE; optional SHAP + exploratory fairness.
6. `make -C research-paper mimic-lock` → `cohort_lock.json` (SHA only).

**Success criteria.**
- [ ] Audit: 0 post-index feature events on integrity path.
- [ ] Manifest + evaluation JSON written; `data_sha256` recorded.
- [ ] Controlled contrast (optional): naive all-event means vs truncate path shows audit failure / metric inflation (same spirit as paper Experiments E/F).
- [ ] No row-level MIMIC data in git or public supplements.

**Out of scope.** Device clearance, prospective deployment, multi-site claims, fabricated AUCs.

**Timeline (indicative).**
| Week | Milestone |
|------|-----------|
| 0–2 | CITI + DUA; local MIMIC load |
| 2–4 | Cohort definition + extract + validate |
| 4–6 | Train/audit/lock; draft methods addendum |
| Ongoing | Collaborator follow-ups every 2–3 weeks |

---

## 4. Patience and persistence

- PhysioNet + local IT + IRB (if required) often dominate calendar time—not the model fit.
- Keep a dated log of CITI completion, DUA signature, download dates, and who you emailed.
- Until lock exists: paper stays on **synthetic verification only**; do not add MIMIC numbers to the manuscript.

---

## Quick commands (after extract exists)

```bash
# Validate
PYTHONPATH=. python scripts/validate_training_data.py \
  --format longitudinal data/processed/mimic_diabetes_cohort.csv

# Train (example)
PYTHONPATH=. python -m training.train \
  --format longitudinal \
  --data data/processed/mimic_diabetes_cohort.csv \
  --model logreg --calibrate \
  --split-by-patient \
  --windows-days 7,30,180 \
  --horizon-days 365 \
  --index-time-col index_time

# Lock (SHA only → research-paper/reports/mimic/)
make -C research-paper mimic-lock
```
