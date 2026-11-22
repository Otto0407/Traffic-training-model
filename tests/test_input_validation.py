import numpy as np
import pandas as pd
import pytest

from features import select_features
from input_validation import (
    PEAK_PERIOD_BY_HOUR,
    WEEKDAY_RUSH_HOURS,
    canonicalise_category,
    known_categories,
    normalise_column_name,
    parse_binary,
    parse_number,
    validate_input,
)


def as_text(df):
    """What predict.read_table produces: every cell as a string."""
    return df.astype(str).replace({"NaT": "", "nan": ""})


@pytest.fixture(scope="module")
def categories(fitted_preprocessor):
    return known_categories(fitted_preprocessor)


def test_calendar_rules_match_the_training_data(clean_df):
    assert (clean_df["peak_period"] == clean_df["hour"].map(PEAK_PERIOD_BY_HOUR)).all()
    rush = (clean_df["hour"].isin(WEEKDAY_RUSH_HOURS) & (clean_df["is_weekend"] == 0)).astype(int)
    assert (clean_df["rush_hour"] == rush).all()
    assert (clean_df["is_weekend"] == (clean_df["day_of_week"] >= 5).astype(int)).all()


@pytest.mark.parametrize("raw,expected", [
    ("Weather", "weather_condition"), (" Zone ", "city_zone"), ("Is Holiday", "is_holiday"),
    ("temp", "temperature"), ("Rain-fall_MM", "rain_fall_mm"), ("hour", "hour"),
])
def test_column_names_are_normalised(raw, expected):
    assert normalise_column_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("3.5 mm", 3.5), ("1,200", 1200.0), ("12,5", 12.5), ("-2", -2.0), ("N/A", np.nan), ("", np.nan),
    ("two", np.nan), ("1e3", 1000.0),
])
def test_numbers_written_as_text(raw, expected):
    value = parse_number(pd.Series([raw]))[0]
    assert (np.isnan(value) and np.isnan(expected)) or value == expected


@pytest.mark.parametrize("raw,expected", [
    ("yes", 1), ("Y", 1), ("true", 1), ("1.0", 1), ("no", 0), ("FALSE", 0), ("0", 0), ("maybe", np.nan), ("", np.nan),
])
def test_binary_flags(raw, expected):
    value = parse_binary(pd.Series([raw]))[0]
    assert (np.isnan(value) and np.isnan(expected)) or value == expected


def test_category_spelling_variants(categories):
    raw = pd.Series([" heavy_rain ", "CLEAR", "Sandstorm", "", "fog"])
    mapped, unknown = canonicalise_category(raw, categories["weather_condition"])
    assert list(mapped[:2]) == ["Heavy Rain", "Clear"] and mapped[4] == "Fog"
    assert pd.isna(mapped[2]) and pd.isna(mapped[3]) and unknown == 1


def test_clean_input_gives_the_same_model_input_as_training(fitted_preprocessor, categories, small_df):
    rows = small_df.sample(400, random_state=0)
    features, quality, report = validate_input(as_text(rows), categories)
    expected = fitted_preprocessor.transform(select_features(rows)[0]).astype(float)
    assert np.allclose(fitted_preprocessor.transform(features).astype(float), expected)
    assert (quality["data_quality"] == "ok").all()
    assert report["unknown_category_values"] == report["out_of_range_values"] == 0


@pytest.fixture
def messy(small_df):
    rows = small_df.sample(12, random_state=1).reset_index(drop=True)
    m = as_text(rows).rename(columns={"weather_condition": "Weather", "city_zone": "Zone ",
                                      "rainfall_mm": "Rainfall", "temperature": "temp", "is_holiday": "Is Holiday"})
    m = m.drop(columns=["parking_occupancy"])
    m["Plate Number"] = [f"ABC-{i:04d}" for i in range(12)]
    m["driver_name"] = "Jane Doe"
    m["Phone"] = "0912-345-678"
    m.loc[0, "Rainfall"] = "3.5 mm"
    m.loc[1, "Rainfall"] = "1,200"           # out of range
    m.loc[2, "temp"] = "N/A"
    m.loc[3, "lanes"] = "two"
    m.loc[4, "Is Holiday"] = "yes"
    m.loc[5, "Is Holiday"] = "No"
    m.loc[6, "Weather"] = " heavy_rain "
    m.loc[7, "Weather"] = "Sandstorm"        # unknown
    m.loc[8, "hour"] = ""                    # rebuilt from timestamp
    m.loc[9, ["hour", "timestamp"]] = ""     # cannot be known
    m.loc[10, "day_of_week"] = "9"           # impossible, rebuilt from timestamp
    m.loc[11, "Zone "] = "downtown core"
    return m, rows


def test_messy_input_is_repaired(messy, categories):
    m, rows = messy
    features, quality, report = validate_input(m, categories)
    assert features.loc[0, "rainfall_mm"] == 3.5
    assert np.isnan(features.loc[1, "rainfall_mm"]) and np.isnan(features.loc[2, "temperature"])
    assert np.isnan(features.loc[3, "lanes"])
    assert features.loc[4, "is_holiday"] == 1 and features.loc[5, "is_holiday"] == 0
    assert features.loc[6, "weather_condition"] == "Heavy Rain" and pd.isna(features.loc[7, "weather_condition"])
    assert features.loc[8, "hour"] == rows.loc[8, "timestamp"].hour
    assert features.loc[10, "day_of_week"] == rows.loc[10, "timestamp"].dayofweek
    assert features.loc[11, "city_zone"] == "Downtown Core"
    assert report["columns_missing"] == ["parking_occupancy"]
    assert report["unknown_category_values"] == 1 and report["out_of_range_values"] == 2
    assert report["calendar_values_derived_from_timestamp"] == 2


def test_quality_flags(messy, categories):
    features, quality, _ = validate_input(messy[0], categories)
    assert quality.loc[9, "data_quality"] == "insufficient_data"
    assert set(quality.drop(index=9)["data_quality"]) == {"imputed"}  # parking_occupancy missing everywhere
    assert (quality["n_features_imputed"] >= 1).all()


def test_many_missing_features_flag_low_quality(small_df, categories):
    row = as_text(small_df.head(1))
    for col in ["temperature", "rainfall_mm", "visibility_km", "lanes", "speed_limit", "weather_condition"]:
        row[col] = ""
    _, quality, _ = validate_input(row, categories)
    assert quality.loc[row.index[0], "data_quality"] == "low_quality"


def test_personal_data_columns_are_dropped_and_never_reported(messy, categories):
    features, _, report = validate_input(messy[0], categories)
    assert not {"plate_number", "driver_name", "phone"} & set(features.columns)
    assert report["columns_dropped_personal_data"] == 5  # plate, driver, phone, latitude, longitude
    text = str(report)
    for secret in ("ABC-0000", "Jane Doe", "0912-345-678"):
        assert secret not in text


def test_input_is_not_modified(messy, categories):
    m = messy[0]
    before = m.copy()
    validate_input(m, categories)
    pd.testing.assert_frame_equal(m, before)
