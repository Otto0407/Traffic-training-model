"""Validate and normalise real-world input before prediction.

Real data arrives with renamed columns, numbers stored as text ("12 mm",
"1,200", "N/A"), yes/no flags, inconsistent spelling of categories, missing
columns and personal data the model does not need. This module turns such a
table into the exact feature columns the model was trained on and reports,
per row, how much had to be filled in.

Privacy: only the model's feature columns (plus a timestamp used to derive
calendar fields) are kept. Every other column is dropped on arrival and never
copied to the output or printed; reports contain counts and column names
only, never cell values.
"""
import re

import numpy as np
import pandas as pd

from config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TIMESTAMP, VALID_RANGES

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES

# Common alternative names for the model's inputs (after normalising case,
# spaces and punctuation). Extend this when a new data source uses other names.
COLUMN_ALIASES = {
    "zone": "city_zone",
    "area": "city_zone",
    "weather": "weather_condition",
    "rain": "rainfall_mm",
    "rainfall": "rainfall_mm",
    "precipitation_mm": "rainfall_mm",
    "visibility": "visibility_km",
    "temp": "temperature",
    "temperature_c": "temperature",
    "weekday": "day_of_week",
    "dow": "day_of_week",
    "speed_limit_kmh": "speed_limit",
    "num_lanes": "lanes",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "time": "timestamp",
}

# Column names that suggest personal or identifying data. Such columns are
# never needed by the model; they are dropped and only their count is reported.
PII_NAME_PATTERN = re.compile(
    r"plate|licen[cs]e|driver|owner|passenger|person|name|phone|mobile|e_?mail|address|"
    r"^ip$|ip_addr|mac_addr|imei|device_id|vin\b|user_id|account|gps|trace|face|image|photo|"
    r"latitude|longitude|^lat$|^lon$|^lng$",
    re.IGNORECASE,
)

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "unknown", "missing"}
TRUE_TOKENS = {"1", "1.0", "true", "t", "yes", "y"}
FALSE_TOKENS = {"0", "0.0", "false", "f", "no", "n"}

# Calendar fields follow fixed rules in the training data (checked by
# tests/test_input_validation.py), so they can be rebuilt from a timestamp.
PEAK_PERIOD_BY_HOUR = {
    **{h: "Night" for h in (0, 1, 2, 3, 4, 5, 22, 23)},
    **{h: "Off-Peak" for h in (6, 10, 11, 15, 20, 21)},
    **{h: "Morning Peak" for h in (7, 8, 9)},
    **{h: "Midday" for h in (12, 13, 14)},
    **{h: "Evening Peak" for h in (16, 17, 18, 19)},
}
WEEKDAY_RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}

# Without the hour the model has almost nothing to go on (removing time
# features drops macro F1 to ~0.33), so such rows are not predicted.
CRITICAL_FEATURES = ("hour",)
# Above this many filled-in features a prediction is flagged as low quality.
MAX_IMPUTED_FOR_OK = 5


def normalise_column_name(name) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return COLUMN_ALIASES.get(name, name)


def _is_missing_token(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)


def parse_number(series: pd.Series) -> pd.Series:
    """'12.5 mm' -> 12.5, '1,200' -> 1200, '12,5' -> 12.5, 'N/A' -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    thousands = text.str.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?")
    text = text.where(~thousands, text.str.replace(",", "", regex=False))
    text = text.str.replace(",", ".", regex=False)
    number = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(number, errors="coerce").mask(_is_missing_token(series))


def parse_binary(series: pd.Series) -> pd.Series:
    """yes/no, true/false, 1/0 (also '1.0') -> 1/0; anything else -> NaN."""
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index)
    out[text.isin(TRUE_TOKENS)] = 1.0
    out[text.isin(FALSE_TOKENS)] = 0.0
    return out


def _category_key(value) -> str:
    return re.sub(r"[\s_\-]+", " ", str(value).strip().lower())


def canonicalise_category(series: pd.Series, known: list) -> tuple:
    """Map spelling variants onto the categories seen in training
    ('heavy_rain', ' HEAVY RAIN ' -> 'Heavy Rain'). Unknown values -> NaN.
    Returns (series, number of unknown values)."""
    if known and all(isinstance(k, (int, np.integer, float)) for k in known):
        values = parse_number(series)
        unknown = values.notna() & ~values.isin(known)
        return values.mask(unknown), int(unknown.sum())
    lookup = {_category_key(k): k for k in known}
    keys = series.map(_category_key)
    mapped = keys.map(lookup)
    missing = _is_missing_token(series)
    unknown = mapped.isna() & ~missing
    return mapped.where(~missing), int(unknown.sum())


def derive_calendar_fields(df: pd.DataFrame) -> int:
    """Fill hour, day_of_week, is_weekend, peak_period and rush_hour from the
    timestamp where they are missing. Returns how many values were derived."""
    if TIMESTAMP not in df:
        return 0
    ts = pd.to_datetime(df[TIMESTAMP], errors="coerce", format="mixed")
    derived = 0

    def fill(col, values):
        nonlocal derived
        if col not in df:
            df[col] = np.nan
        gap = df[col].isna() & values.notna()
        derived += int(gap.sum())
        df.loc[gap, col] = values[gap]

    fill("hour", ts.dt.hour.astype(float))
    fill("day_of_week", ts.dt.dayofweek.astype(float))
    fill("is_weekend", (df["day_of_week"] >= 5).astype(float).where(df["day_of_week"].notna()))
    fill("peak_period", df["hour"].map(PEAK_PERIOD_BY_HOUR))
    rush = df["hour"].isin(WEEKDAY_RUSH_HOURS) & (df["is_weekend"] == 0)
    fill("rush_hour", rush.astype(float).where(df["hour"].notna() & df["is_weekend"].notna()))
    return derived


def known_categories(preprocessor) -> dict:
    """Categories each categorical column had in training, read from the
    fitted one-hot encoder, so the input can be checked against them."""
    encoder = preprocessor.named_transformers_["cat"][-1]
    return {col: list(cats) for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)}


def validate_input(raw: pd.DataFrame, categories: dict) -> tuple:
    """Return (features, row_quality, report).

    features: one row per input row, exactly the model's feature columns,
        with problems turned into NaN for the preprocessor to fill in.
    row_quality: per row, the number of filled-in features and a status of
        "ok", "imputed", "low_quality" or "insufficient_data".
    report: counts only (no cell values), safe to print or store.
    """
    report = {"rows": len(raw)}
    df = raw.copy()
    df.columns = [normalise_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    keep = set(FEATURES) | {TIMESTAMP}
    dropped = [c for c in df.columns if c not in keep]
    report["columns_dropped_personal_data"] = sum(bool(PII_NAME_PATTERN.search(c)) for c in dropped)
    report["columns_dropped_other"] = len(dropped) - report["columns_dropped_personal_data"]
    df = df[[c for c in df.columns if c in keep]]

    report["columns_missing"] = sorted(c for c in FEATURES if c not in df)
    for col in FEATURES:
        if col not in df:
            df[col] = np.nan

    for col in NUMERICAL_FEATURES:
        df[col] = parse_number(df[col])
    for col in BINARY_FEATURES:
        df[col] = parse_binary(df[col])
    unknown = 0
    for col in CATEGORICAL_FEATURES:
        if col in ("hour", "day_of_week"):
            df[col] = parse_number(df[col])
        else:
            df[col], n = canonicalise_category(df[col], categories.get(col, []))
            unknown += n
    report["unknown_category_values"] = unknown

    out_of_range = 0
    for col, (low, high) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < low) | (df[col] > high))
        out_of_range += int(bad.sum())
        df[col] = df[col].mask(bad)
    for col in ("hour", "day_of_week"):
        bad = df[col].notna() & (df[col] % 1 != 0)
        out_of_range += int(bad.sum())
        df[col] = df[col].mask(bad)
    report["out_of_range_values"] = out_of_range

    report["calendar_values_derived_from_timestamp"] = derive_calendar_fields(df)
    for col in ("hour", "day_of_week"):
        df[col], _ = canonicalise_category(df[col], categories.get(col, []))

    features = df[FEATURES]
    n_missing = features.isna().sum(axis=1)
    critical = features[list(CRITICAL_FEATURES)].isna().any(axis=1)
    status = np.select(
        [critical, n_missing > MAX_IMPUTED_FOR_OK, n_missing > 0],
        ["insufficient_data", "low_quality", "imputed"],
        default="ok",
    )
    quality = pd.DataFrame({"n_features_imputed": n_missing.to_numpy(), "data_quality": status}, index=raw.index)
    report["rows_by_quality"] = quality["data_quality"].value_counts().to_dict()
    return features, quality, report
