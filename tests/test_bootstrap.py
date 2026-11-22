import numpy as np

from bootstrap import bootstrap_ci


def accuracy(y, p):
    return float((y == p).mean())


def test_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 2000)
    p = np.where(rng.random(2000) < 0.8, y, (y + 1) % 4)
    groups = np.repeat(np.arange(200), 10)
    point, low, high = bootstrap_ci(y, p, groups, accuracy, n_boot=300)
    assert low <= point <= high and 0.75 < point < 0.85


def test_same_seed_gives_same_interval():
    y = np.arange(100) % 4
    p = np.roll(y, 1)
    groups = np.arange(100) // 5
    assert bootstrap_ci(y, p, groups, accuracy, 200, seed=3) == bootstrap_ci(y, p, groups, accuracy, 200, seed=3)


def test_constant_metric_gives_zero_width():
    y = np.zeros(50, dtype=int)
    _, low, high = bootstrap_ci(y, y, np.arange(50), accuracy, n_boot=100)
    assert low == high == 1.0


def test_group_resampling_is_wider_than_row_resampling_for_correlated_errors():
    # Errors come in whole groups (like all intersections in one bad hour).
    y = np.zeros(1000, dtype=int)
    p = np.repeat(np.random.default_rng(1).integers(0, 2, 100), 10)
    _, lo_g, hi_g = bootstrap_ci(y, p, np.repeat(np.arange(100), 10), accuracy, 400)
    _, lo_r, hi_r = bootstrap_ci(y, p, np.arange(1000), accuracy, 400)
    assert (hi_g - lo_g) > 1.5 * (hi_r - lo_r)
