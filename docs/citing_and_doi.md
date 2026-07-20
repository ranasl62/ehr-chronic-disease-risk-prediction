# Citing this software and obtaining a DOI

Researchers should cite the **software artifact** when they use the EHR Risk Framework pipeline (and cite any future journal paper separately once published).

## Quick cite

1. On GitHub, open this repository → **Cite this repository** (uses [`CITATION.cff`](../CITATION.cff)).
2. Or copy the BibTeX below.

```bibtex
@software{ehr_risk_framework_hossain_2026,
  author       = {Hossain, Md Rana},
  title        = {{EHR Risk Framework}: Leakage-Aware, Calibrated, Explainable Open Software},
  version      = {1.0.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21448693},
  url          = {https://doi.org/10.5281/zenodo.21448693},
  license      = {MIT}
}
```

**DOI:** [https://doi.org/10.5281/zenodo.21448693](https://doi.org/10.5281/zenodo.21448693) (`10.5281/zenodo.21448693`).

## Zenodo DOI

This repository is archived on Zenodo:

- **DOI:** [https://doi.org/10.5281/zenodo.21448693](https://doi.org/10.5281/zenodo.21448693)
- **Identifier:** `10.5281/zenodo.21448693`

Citation metadata lives in [`CITATION.cff`](../CITATION.cff), [`.zenodo.json`](../.zenodo.json), and the README “Cite this repository” section.

For later versions: create a new GitHub Release with Zenodo–GitHub still enabled; Zenodo mints a version DOI while the concept DOI stays stable. Do **not** invent DOIs. Release hygiene: [`release_checklist.md`](release_checklist.md).

## Paper vs software

| Artifact | When to cite |
|----------|----------------|
| This GitHub / Zenodo software DOI | You used the code, Docker images, or the local academic verification pipeline (`make -C research-paper paper-quick`) |
| Future journal article DOI | You rely on the peer-reviewed methods narrative |

Do not treat synthetic verification AUCs in `research-paper/reports/` as clinical performance when citing.

## License

MIT — see [`LICENSE`](../LICENSE). Author metadata: [`AUTHORS.md`](../AUTHORS.md).
