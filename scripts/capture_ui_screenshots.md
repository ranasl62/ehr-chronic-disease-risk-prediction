# Capture UI screenshots

Prefer the automated script (stack must be up):

```bash
docker compose up -d
bash scripts/capture_docs_website_screenshots.sh
```

PNGs land in `docs/media/` (`01_` … `09_`). Tour: `docs/workbench/`.

## Hygiene

- Demo / synthetic data only — never PHI or credentialed MIMIC rows
- Crop OS usernames and private hostnames before committing
