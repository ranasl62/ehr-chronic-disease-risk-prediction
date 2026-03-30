import numpy as np

from training.calibration_metrics import expected_calibration_error


def test_ece_perfect_on_constant_prob():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    ece = expected_calibration_error(y, p, n_bins=5)
    assert ece < 0.01


def test_ece_finite():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.random(size=100)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0 <= ece <= 1
