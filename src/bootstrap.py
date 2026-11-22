"""Bootstrap confidence intervals for test-set metrics."""
import numpy as np


def bootstrap_ci(y_true, y_pred, groups, metric_fn, n_boot: int = 1000, level: float = 0.95, seed: int = 0) -> tuple:
    """Return (point estimate, lower, upper) for metric_fn(y_true, y_pred).

    Resamples whole groups (here: test hours) rather than single rows. All 100
    intersections in the same hour share the weather and calendar, so their
    errors are correlated; resampling rows independently would make the
    interval too narrow.
    """
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([rows_by_group[g] for g in picked])
        scores[b] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1 - level) / 2
    lower, upper = np.quantile(scores, [alpha, 1 - alpha])
    return metric_fn(y_true, y_pred), float(lower), float(upper)
