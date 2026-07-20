# Documentation website (`docs/`)

Public site for the **EHR Risk Framework** — students, researchers, and health-informatics labs.

## Local preview

```bash
cd docs
python3 -m http.server 4173
# open http://127.0.0.1:4173/
```

## GitHub Pages (point reviewers here)

Publish from this folder as the site root (so `index.html` is at `/`):

- **Branch → `/docs`**, or
- **GitHub Actions** via `.github/workflows/pages.yml` (upload path `docs`)

Custom domain: `https://ehr.larucare.com/` (see `CNAME`).

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
| `limits.html` / `help.html` / `cite.html` | Boundaries, full guide library, contact |

Navigation, feedback strip, continue-reading, and footer map: `assets/site.js`.

## Framing

Research and education only. Do not imply clinical deployment or regulated medical-device status.
U.S. healthcare research context on Home / Why describes public scientific value — not a product claim.
