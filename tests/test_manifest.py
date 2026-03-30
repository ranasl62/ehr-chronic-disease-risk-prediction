from pathlib import Path

from training.manifest import build_training_manifest, sha256_file


def test_sha256_stable(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    assert len(sha256_file(p)) == 64


def test_manifest_keys(tmp_path: Path):
    d = tmp_path / "d.csv"
    d.write_text("a,b\n1,2\n", encoding="utf-8")
    m = tmp_path / "m.pkl"
    m.write_bytes(b"x")
    man = build_training_manifest(
        data_path=d,
        model_path=m,
        model_kind="logreg",
        calibrated=False,
        split_method="patient_group",
    )
    assert man["data_sha256"]
    assert man["split_method"] == "patient_group"
