import numpy as np
import pytest
import torch

from evaluate import compute_metrics, predict
from model import CongestionMLP


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 0, 3])
    m = compute_metrics(y, y)
    assert m["accuracy"] == m["f1_macro"] == m["qwk"] == 1.0
    assert m["mean_level_error"] == 0 and m["errors_two_or_more_levels"] == 0


def test_ordinal_metrics_punish_distant_errors_more():
    y = np.array([0, 0, 0, 3, 3, 3, 1, 2])
    near = y.copy(); near[0] = 1        # Low predicted as Moderate
    far = y.copy(); far[0] = 3          # Low predicted as Severe
    m_near, m_far = compute_metrics(y, near), compute_metrics(y, far)
    assert m_near["f1_macro"] != 1 and m_far["f1_macro"] != 1
    assert m_far["qwk"] < m_near["qwk"]
    assert m_far["mean_level_error"] == 3 * m_near["mean_level_error"]
    assert m_far["errors_two_or_more_levels"] == 1 and m_near["errors_two_or_more_levels"] == 0


def test_confusion_matrix_has_all_four_classes_even_if_absent():
    m = compute_metrics(np.array([0, 0]), np.array([0, 1]))
    assert np.array(m["confusion_matrix"]).shape == (4, 4)


def test_predict_returns_class_indices_in_batches(cpu):
    model = CongestionMLP(5, [8], 4, 0.0)
    X = np.random.rand(10_000, 5).astype(np.float32)
    preds = predict(model, X, cpu, batch_size=4096)
    assert preds.shape == (10_000,) and set(np.unique(preds)) <= {0, 1, 2, 3}
