from utils.eval_report import evaluation_aligned_with_manifest


def test_alignment_requires_matching_sha():
    assert not evaluation_aligned_with_manifest(None, {})
    assert not evaluation_aligned_with_manifest({"data_sha256": "a"}, {})
    assert evaluation_aligned_with_manifest(
        {"data_sha256": "x"},
        {"training_manifest": {"data_sha256": "x"}},
    )
    assert not evaluation_aligned_with_manifest(
        {"data_sha256": "x"},
        {"training_manifest": {"data_sha256": "y"}},
    )
