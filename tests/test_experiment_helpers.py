import numpy as np
import pandas as pd

from experiments import EXPERIMENTS, feature_sets, metric_row, summarise
from tune import DEFAULT_LABELS, SEARCH


def test_every_experiment_refers_to_a_known_feature_set(clean_df):
    sets = feature_sets(clean_df)
    assert all(fs_key in sets for _, _, fs_key, _, _ in EXPERIMENTS)
    assert len({name for name, *_ in EXPERIMENTS}) == len(EXPERIMENTS)


def test_metric_row_and_summary():
    y = np.array([0, 1, 2, 3] * 5)
    rows = [metric_row("a", s, y, y, y_val=y, val_pred=y) for s in (1, 2)]
    rows.append(metric_row("b", 1, y, np.roll(y, 1), y_val=y, val_pred=np.roll(y, 1)))
    summary = summarise(pd.DataFrame(rows))
    assert list(summary.index) == ["a", "b"]
    assert summary.loc["a", "macro_f1"] == "1.000" and summary.loc["a", "val_macro_f1"] == "1.000"


def test_tuning_search_space_is_consistent():
    assert set(DEFAULT_LABELS) == {p for p, _ in SEARCH}
    for param, options in SEARCH:
        labels = [label for label, _ in options]
        assert DEFAULT_LABELS[param] not in labels and len(labels) == len(set(labels))
