# Documentation website (`docs/`)

Public site for the **EHR Risk Framework** — students, researchers, and health-informatics labs.

## Local preview

```bash
cd docs
python3 -m http.server 4173
# open http://127.0.0.1:4173/
```

Clean URLs use the folder/`index.html` pattern (GitHub Pages native):

| Live URL | File |
|----------|------|
| `/` | `index.html` |
| `/quickstart` | `quickstart/index.html` |
| `/why` | `why/index.html` |
| … | … |

## GitHub Pages (point reviewers here)

Publish from this folder as the site root (so `index.html` is at `/`):

- **Branch → `/docs`**, or
- **GitHub Actions** via `.github/workflows/pages.yml` (upload path `docs`)

Custom domain: `https://ehr.larucare.com/` (see `CNAME`).

Internal links use site-root absolute paths (`/`, `/quickstart`, …) so the custom domain and project-pages URLs stay consistent.

## Screenshots

```bash
# stack on :8080 / :8000
bash scripts/capture_docs_website_screenshots.sh
```

Demo data only — never capture PHI.

## Pages

| Path | Content |
|------|---------|
| `/` | Landing, U.S. healthcare research context, capabilities |
| `/why` | Problems, audiences, U.S. research relevance |
| `/features` | Feature catalog |
| `/workbench` | Hub — every UI route + screenshots |
| `/ui-*` | Per-page UI detail (9 pages) |
| `/sitemap` | Full map |
| `/diagrams` · `/fine-tuning` | Design + iteration |
| `/quickstart` · `/commands` · `/docker-images` | Run recipes |
| `/architecture` · `/api` · `/data` | Deep technical |
| `/limits` · `/help` · `/cite` | Boundaries, full guide library, contact |

Navigation, feedback strip, continue-reading, and footer map: `assets/site.js`.

## Framing

For research and education only. Outputs are not clinical recommendations and are not intended for patient care. We are working toward broader general-purpose use in the future. Do not imply clinical deployment or regulated medical-device status.
U.S. healthcare research context on Home / Why describes public scientific value — not a product claim.
