import pandas as pd
import pytest

from cleaning import clean_data, format_report
from config import TIMESTAMP


@pytest.fixture
def dirty(raw_data):
    d = raw_data.head(300).copy()
    d = pd.concat([d, d.iloc[:5]])                        # 5 exact duplicates
    d.loc[d.index[10], "congestion_level"] = "Unknown"     # invalid label
    d.loc[d.index[11], "congestion_level"] = " Low "       # whitespace -> fixed
    d.loc[d.index[12], TIMESTAMP] = pd.NaT                 # missing time
    d.loc[d.index[13], "temperature"] = -100000            # impossible value
    d.loc[d.index[14], "is_weekend"] = 7                   # invalid flag
    d.loc[d.index[15], "weather_condition"] = "  "         # blank category
    return d.reset_index(drop=True)


def test_real_data_passes_through_unchanged(raw_data):
    clean, report = clean_data(raw_data)
    assert report["rows_out"] == report["rows_in"] == len(raw_data)
    assert (clean.dtypes == raw_data.dtypes).all()
    # Row order must be kept: re-sorting changes mini-batches and results.
    assert (clean["record_id"].to_numpy() == raw_data["record_id"].to_numpy()).all()


def test_dirty_rows_are_fixed_or_dropped(dirty):
    clean, report = clean_data(dirty)
    assert report["duplicate_rows_dropped"] == 5
    assert report["invalid_label_rows_dropped"] == 1
    assert report["missing_timestamp_rows_dropped"] == 1
    assert report["out_of_range_values_set_missing"] == 1
    assert report["invalid_binary_values_set_missing"] == 1
    assert report["rows_out"] == 298
    assert "Low" in clean["congestion_level"].values and " Low " not in clean["congestion_level"].values
    assert clean["temperature"].isna().sum() == 1
    assert clean["weather_condition"].isna().sum() == 1


def test_repeated_intersection_hour_keeps_first(raw_data):
    d = raw_data.head(5).copy()
    twin = d.iloc[[0]].copy()
    twin["temperature"] = 99.0  # same intersection and hour, different reading
    clean, report = clean_data(pd.concat([d, twin]))
    assert report["duplicate_intersection_hours_dropped"] == 1
    assert clean.iloc[0]["temperature"] == d.iloc[0]["temperature"]


def test_cleaning_is_idempotent(dirty):
    once, _ = clean_data(dirty)
    twice, report = clean_data(once)
    assert report["rows_out"] == report["rows_in"]
    pd.testing.assert_frame_equal(once, twice)


def test_input_is_not_modified(dirty):
    before = dirty.copy()
    clean_data(dirty)
    pd.testing.assert_frame_equal(dirty, before)


def test_report_formats_every_key(dirty):
    _, report = clean_data(dirty)
    assert len(format_report(report).splitlines()) == len(report)
