# Release checklist (v1.0.0 and later)

Prepare a citeable GitHub release of this **framework**. Do **not** include PhysioNet/MIMIC row data.

## Before tagging

- [ ] `CHANGELOG.md` section matches this release
- [ ] `setup.py` / `openhealth.__version__` / `CITATION.cff` version strings agree
- [ ] `PYTHONPATH=. pytest tests/ -q` green on `main`
- [ ] README badges and benchmark table accurate (MIMIC row still *pending lock* unless `cohort_lock.json` exists)
- [ ] No PHI under `data/processed/`, `screenshots/`, or release assets

## Tag and GitHub Release

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

Create a GitHub Release from the tag; attach nothing that contains restricted EHR extracts. Point release notes at `CHANGELOG.md`.

## Optional Zenodo DOI

See full steps: [`citing_and_doi.md`](citing_and_doi.md).

1. Connect the GitHub repo to [Zenodo](https://zenodo.org/) (GitHub app → enable this repository).
2. Publish a GitHub Release from an annotated tag; Zenodo archives it and mints a DOI.
3. Copy the real DOI (do **not** invent one). Example format only: `10.5281/zenodo.XXXXXXX`.
4. Uncomment and fill in [`CITATION.cff`](../CITATION.cff):

```yaml
identifiers:
  - type: doi
    value: "10.5281/zenodo.XXXXXXX"
```

5. Update the Cite section in [`README.md`](../README.md) and optionally [`.zenodo.json`](../.zenodo.json).

## Optional next steps

- [ ] Publish to PyPI (confirm package name availability)
- [ ] Credentialed MIMIC evaluation via [`mimic_lock_checklist.md`](mimic_lock_checklist.md) (local only; never commit extracts)
