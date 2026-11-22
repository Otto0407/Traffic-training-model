import numpy as np
import pytest

from config import BINARY_FEATURES, CATEGORICAL_FEATURES, DROPPED_FEATURES, NUMERICAL_FEATURES, TARGET, TIMESTAMP
from features import LEAK_FREE, VEHICLE_RATIOS, add_vehicle_ratios, all_columns_feature_set, select_features

KEPT = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES
DROPPED = [c for cols in DROPPED_FEATURES.values() for c in cols]

# Columns that reveal the label; none may ever reach the model.
LEAKING = ["congestion_score", "traffic_density", "vehicle_count", "average_speed", "queue_length",
           "emission_estimate", "fuel_waste_estimate", "green_light_duration", "signal_cycle_seconds"]


def test_every_column_is_kept_or_dropped_exactly_once(raw_data):
    assert len(KEPT) == len(set(KEPT)) == 25
    assert len(DROPPED) == len(set(DROPPED))
    assert not set(KEPT) & set(DROPPED)
    assert set(KEPT) | set(DROPPED) | {TARGET} == set(raw_data.columns)


@pytest.mark.parametrize("column", LEAKING)
def test_leaking_columns_are_excluded(column):
    assert column not in LEAK_FREE.columns


def test_select_features_returns_kept_columns_and_label(clean_df):
    X, y = select_features(clean_df.head(10))
    assert list(X.columns) == LEAK_FREE.columns
    assert y.name == TARGET and len(y) == 10


def test_unaccounted_column_is_refused(clean_df):
    with pytest.raises(ValueError, match="neither kept nor dropped"):
        select_features(clean_df.head(5).assign(new_sensor=1))


def test_missing_column_is_refused(clean_df):
    with pytest.raises(ValueError, match="missing"):
        select_features(clean_df.head(5).drop(columns=["hour"]))


def test_feature_set_without():
    fs = LEAK_FREE.without("hour", "lanes", "is_weekend")
    assert not {"hour", "lanes", "is_weekend"} & set(fs.columns)
    assert len(fs.columns) == len(LEAK_FREE.columns) - 3


def test_all_columns_set_excludes_ids_time_and_label(clean_df):
    cols = all_columns_feature_set(clean_df).columns
    assert not {"record_id", TIMESTAMP, TARGET} & set(cols)
    assert "congestion_score" in cols


def test_vehicle_ratios_sum_to_one(clean_df):
    df = add_vehicle_ratios(clean_df.head(200))
    assert np.allclose(df[list(VEHICLE_RATIOS)].sum(axis=1), 1.0)
    assert "heavy_vehicle_ratio" not in clean_df  # input not modified
