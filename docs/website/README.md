# Documentation website (`docs/website/`)

Public site for the **EHR Risk Framework** — students, researchers, and health-informatics labs.

## Local preview

```bash
cd docs/website
python3 -m http.server 4173
# open http://127.0.0.1:4173/
```

## GitHub Pages (point reviewers here)

1. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
2. Push to `main` (or run **GitHub Pages (docs site)** manually)
3. Typical URL: `https://ranasl62.github.io/ehr-chronic-disease-risk-prediction/`

Workflow: `.github/workflows/pages.yml` publishes this folder.

Relative links are used throughout so project-pages URLs work without a custom base href.

## Screenshots

```bash
# stack on :8080 / :8000
bash scripts/capture_docs_website_screenshots.sh
```

Demo data only — never capture PHI.

## Pages

| File | Content |
|------|---------|
| `index.html` | Landing, U.S. healthcare research context, capabilities |
| `why.html` | Problems, audiences, U.S. research relevance |
| `features.html` | Feature catalog |
| `workbench.html` | Hub — every UI route + screenshots |
| `ui-*.html` | Per-page UI detail (9 pages) |
| `sitemap.html` | Full map |
| `diagrams.html` / `fine-tuning.html` | Design + iteration |
| `quickstart.html` / `commands.html` | Run recipes |
| `architecture.html` / `api.html` / `data.html` | Deep technical |
| `limits.html` / `cite.html` | Boundaries + contact |

Navigation, feedback strip, continue-reading, and footer map: `assets/site.js`.

## Framing

Research and education only. Do not imply clinical deployment or regulated medical-device status.
U.S. healthcare research context on Home / Why describes public scientific value — not a product claim.
