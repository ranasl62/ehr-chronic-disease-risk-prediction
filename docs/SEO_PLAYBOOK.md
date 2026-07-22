# SEO playbook — EHR Risk Framework

Canonical docs site: **https://ehr.larucare.com/**  
Live demo (workbench): **https://ehr-risk-framework.larucare.com/**  
Live API: **https://ehr-api.larucare.com/**  
GitHub: https://github.com/ranasl62/ehr-chronic-disease-risk-prediction  

Catchphrase (use as tagline / anchor text):  
**An open-source framework for leakage-safe, calibrated, and explainable EHR risk prediction.**

Research / education only — not a medical device. Do not invent clinical performance claims.

## Done in-repo (on-page)

- Keyword-rich titles/descriptions via `docs/assets/seo.js` (JSON-LD, Open Graph, canonical)
- `robots.txt` + `sitemap.xml`
- Social card: `docs/assets/og-card.png` (also use as GitHub repo social preview)
- Blog + compare + alternatives + listicle pages under `docs/`
- README catchphrase, problem statement, alternatives, website + live demo links
- Docs home CTA → live demo at `ehr-risk-framework.larucare.com`

## GitHub settings (manual)

1. **Website:** Repository → About → Website → `https://ehr.larucare.com/`
2. **Description (suggested):**  
   `Open-source workbench for leakage-safe, calibrated, explainable EHR risk prediction — Docker + FastAPI + Angular. Research/education only.`
3. **Topics (add 15–20):**  
   `ehr` `healthcare-ai` `clinical-risk-prediction` `clinical-machine-learning` `electronic-health-records` `machine-learning` `python` `fastapi` `angular` `docker` `shap` `calibration` `leakage-detection` `temporal-validation` `health-informatics` `open-source` `research-software` `mimic-iv` `fairness` `risk-prediction`
4. **Social preview:** Settings → Social preview → upload `docs/assets/og-card.png`
5. Optional: `gh repo edit --homepage "https://ehr.larucare.com/" --add-topic ehr --add-topic healthcare-ai …`

## Off-page (ongoing)

| Channel | Action |
|--------|--------|
| Awesome lists | PR to Awesome Health IT / ML / Open Source lists with catchphrase + site link |
| DEV.to / Medium | Guest posts on leakage in clinical AI; link docs + GitHub |
| LinkedIn / X | Project profile; share blog posts with `ehr.larucare.com` |
| Directories | Submit to open-source / research-software catalogs |
| Courses | Ask instructors using the workbench to cite and link the site |

## Keywords to keep natural

- EHR risk prediction  
- clinical machine learning  
- leakage-safe AI / data leakage clinical AI  
- calibrated risk models / ECE / Brier  
- SHAP explainability  
- health informatics research  

Avoid: invented AUCs, “FDA-ready”, “diagnostic”, competitor defamation.
