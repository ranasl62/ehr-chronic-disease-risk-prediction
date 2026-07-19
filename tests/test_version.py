"""Package version alignment for release exhibits."""

from openhealth import __version__


def test_openhealth_version_matches_release():
    assert __version__ == "1.0.0"
