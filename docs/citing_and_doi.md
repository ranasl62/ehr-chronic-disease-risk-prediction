# Citing this software and obtaining a DOI

Researchers should cite the **software artifact** when they use the EHR Risk Framework pipeline (and cite any future journal paper separately once published).

## Quick cite (before Zenodo DOI)

1. On GitHub, open this repository → **Cite this repository** (uses [`CITATION.cff`](../CITATION.cff)).
2. Or copy the BibTeX below.

```bibtex
@software{ehr_risk_framework_hossain_2026,
  author  = {Hossain, Md Rana},
  title   = {{EHR Risk Framework}: Leakage-Aware, Calibrated, Explainable Open Software},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/ranasl62/ehr-chronic-disease-risk-prediction},
  license = {MIT},
  note    = {Add Zenodo DOI after first GitHub Release archive}
}
```

**DOI status:** PENDING until you create a GitHub Release with Zenodo linked. Example format only (not a real DOI for this project): `https://doi.org/10.5281/zenodo.XXXXXXX`.

## Mint a Zenodo DOI (GitHub → Zenodo)

Do **not** invent a DOI. Follow these steps once:

1. Create/sign in to [Zenodo](https://zenodo.org/) (or CERN Sandbox for a dry run).
2. Connect GitHub: Zenodo → **GitHub** → enable the repository `ranasl62/ehr-chronic-disease-risk-prediction`.
3. On GitHub, create a **Release** (e.g. tag `v1.0.0`) with release notes. Zenodo archives the tag and mints a DOI.
4. Copy the **Concept DOI** (version-independent) and/or **Version DOI** from the Zenodo record.
5. Update tracked citation metadata:
   - Uncomment `identifiers` in [`CITATION.cff`](../CITATION.cff) and set `value` to the real `10.5281/zenodo.…` id.
   - Optionally add the DOI to [`.zenodo.json`](../.zenodo.json) related identifiers and to the README “Cite this repository” section.
6. Commit and push those citation updates on a follow-up commit (after the Release is public).

Release hygiene checklist: [`release_checklist.md`](release_checklist.md).

## Paper vs software

| Artifact | When to cite |
|----------|----------------|
| This GitHub / Zenodo software DOI | You used the code, Docker images, or `make paper-quick` pipeline |
| Future journal article DOI | You rely on the peer-reviewed methods narrative |

Do not treat synthetic verification AUCs in `reports/paper/` as clinical performance when citing.

## License

MIT — see [`LICENSE`](../LICENSE). Author metadata: [`AUTHORS.md`](../AUTHORS.md).
