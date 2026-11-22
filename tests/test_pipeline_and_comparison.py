from dataclasses import replace

import numpy as np
import pytest

from compare_models import baseline_predictions, compare
from config import TrainConfig
from pipeline import prepare_data
from reference_models import fit_lightgbm, fit_logistic_regression


def test_prepare_data_shapes_and_frames(prepared, small_df):
    total = len(prepared.y_train) + len(prepared.y_val) + len(prepared.y_test)
    assert total == len(small_df)
    assert len(prepared.train_df) == len(prepared.X_train) == len(prepared.y_train)
    assert prepared.X_val.dtype == np.float32


def test_prepare_data_rejects_unknown_split(small_df):
    with pytest.raises(ValueError, match="Unknown split"):
        prepare_data(small_df, TrainConfig(), split="by_magic")


def test_random_split_option(small_df):
    data = prepare_data(small_df, replace(TrainConfig(), seed=1), split="random")
    assert len(data.y_test) == pytest.approx(0.2 * len(small_df), rel=0.01)


@pytest.mark.slow
def test_reference_models_learn(prepared):
    for fit in (fit_logistic_regression, fit_lightgbm):
        model = fit(prepared, seed=0)
        acc = (model.predict(prepared.X_val) == prepared.y_val).mean()
        assert acc > 0.9, fit.__name__


def test_compare_ranks_by_validation_and_reports_paired_differences(prepared):
    preds = baseline_predictions(prepared)
    preds["Perfect"] = (prepared.y_val, prepared.y_test)
    table, details = compare(prepared, preds, reference="Perfect", n_boot=50)
    assert table.index[0] == "Perfect"
    assert table.loc["Perfect", "test_macro_f1"].startswith("1.000")
    paired = details["paired_macro_f1_difference_vs_Perfect"]
    assert set(paired) == {"Majority class", "Lookup: hour x weekend x zone"}
    assert all(d["difference"] > 0 and d["significant"] for d in paired.values())
