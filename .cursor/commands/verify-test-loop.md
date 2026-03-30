# verify-test-loop

Run the local verify loop: install deps if needed, pytest, and one import smoke.

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
pip install -q -r requirements.txt 2>/dev/null || true
PYTHONPATH=. pytest tests/ -q --tb=short
PYTHONPATH=. python -c "import api.main; import training.train"
```

If pytest fails, fix the smallest failing unit first, re-run, then broaden.
