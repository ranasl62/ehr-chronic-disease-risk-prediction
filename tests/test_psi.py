import numpy as np

from monitoring.psi import psi


def test_psi_identical_low():
    x = np.random.default_rng(0).normal(size=500)
    p = psi(x, x)
    assert p < 0.01
