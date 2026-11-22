import pandas as pd
import pytest

from config import CLASS_ORDER, TARGET, TIMESTAMP
from data_loader import load_data


def test_reads_the_full_dataset(raw_data):
    assert raw_data.shape == (204000, 47)
    assert set(raw_data[TARGET].unique()) == set(CLASS_ORDER)


def test_timestamps_are_parsed_and_sorted(raw_data):
    assert pd.api.types.is_datetime64_any_dtype(raw_data[TIMESTAMP])
    assert raw_data[TIMESTAMP].is_monotonic_increasing


def test_falls_back_to_zip_when_csv_is_missing(tmp_path, raw_data):
    raw_data.head(50).to_csv(tmp_path / "data.zip", index=False, compression="zip")
    df = load_data(tmp_path / "data.csv")
    assert len(df) == 50


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "nothing.csv")


def test_unparseable_timestamp_becomes_nat(tmp_path, raw_data):
    sample = raw_data.head(3).copy()
    sample[TIMESTAMP] = sample[TIMESTAMP].astype(str)
    sample.loc[sample.index[1], TIMESTAMP] = "not a date"
    sample.to_csv(tmp_path / "d.csv", index=False)
    df = load_data(tmp_path / "d.csv")
    assert df[TIMESTAMP].isna().sum() == 1
