#!/usr/bin/env bash
# Capture Angular + OpenAPI screenshots into docs/media/
# Requires: stack on :8080 / :8000, google-chrome or chromium.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEDIA="$ROOT/docs/media"
mkdir -p "$MEDIA"
CHROME="$(command -v google-chrome || command -v chromium-browser || command -v chromium || true)"
if [[ -z "$CHROME" ]]; then
  echo "Chrome/Chromium not found" >&2
  exit 1
fi

shot() {
  local url="$1" out="$2"
  echo "→ $url"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1400,900 \
    --virtual-time-budget=12000 --run-all-compositor-stages-before-draw \
    --screenshot="$out" "$url" >/dev/null 2>&1 || true
  echo "  $(wc -c < "$out") bytes → $out"
}

shot "http://127.0.0.1:8080/" "$MEDIA/01_home_checklist.png"
shot "http://127.0.0.1:8080/datasets" "$MEDIA/02_datasets_health.png"
shot "http://127.0.0.1:8080/config" "$MEDIA/03_config_center.png"
shot "http://127.0.0.1:8080/train" "$MEDIA/04_train_compare.png"
shot "http://127.0.0.1:8080/results" "$MEDIA/05_results_zip.png"
shot "http://127.0.0.1:8080/predict" "$MEDIA/06_predict_banner.png"
shot "http://127.0.0.1:8000/docs" "$MEDIA/07_api_docs.png"
shot "http://127.0.0.1:8080/analytics" "$MEDIA/08_analytics_dashboard.png"
shot "http://127.0.0.1:8080/docs" "$MEDIA/09_ui_docs.png"
shot "http://127.0.0.1:8080/research" "$MEDIA/10_research_wizard.png"

# Legacy fallback removed — re-run capture if 07_api_docs.png is empty
if [[ "$(wc -c < "$MEDIA/07_api_docs.png")" -lt 30000 ]]; then
  echo "warning: 07_api_docs.png looks empty; check :8000/docs is up" >&2
fi
echo "Done. Preview: cd docs && python3 -m http.server 4173"
