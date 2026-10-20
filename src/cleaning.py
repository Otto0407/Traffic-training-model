"""Row-level data cleaning, applied to the whole dataset before splitting.

Only steps that do not learn anything from the data happen here (dropping
duplicates and unusable rows, flagging impossible values). Filling in the
resulting missing values is learned from the training set inside the
preprocessor, so no information from validation or test leaks into it.
"""
import numpy as np
import pandas as pd

from config import BINARY_FEATURES, CATEGORICAL_FEATURES, CLASS_ORDER, TARGET, TIMESTAMP, VALID_RANGES


def clean_data(df: pd.DataFrame) -> tuple:
    """Return (cleaned DataFrame, report) where report counts each fix."""
    df = df.copy()
    report = {"rows_in": len(df)}

    # 1. Exact duplicate rows, then repeated readings for the same
    #    intersection and hour (keep the first).
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["intersection_id", TIMESTAMP], keep="first")
    report["duplicate_intersection_hours_dropped"] = before - len(df)

    # 2. Rows that cannot be used: no valid label or no valid timestamp
    #    (the time-based split needs the timestamp).
    if df[TARGET].dtype == object or pd.api.types.is_string_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].str.strip()
    bad_label = ~df[TARGET].isin(CLASS_ORDER)
    bad_time = df[TIMESTAMP].isna()
    report["invalid_label_rows_dropped"] = int(bad_label.sum())
    report["missing_timestamp_rows_dropped"] = int((bad_time & ~bad_label).sum())
    df = df[~bad_label & ~bad_time]

    # 3. Text categories: trim whitespace and turn empty strings into missing.
    for col in CATEGORICAL_FEATURES:
        if col in df and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            df[col] = df[col].str.strip().replace("", np.nan)

    # 4. Impossible values become missing (filled in later by the preprocessor).
    out_of_range = 0
    for col, (low, high) in VALID_RANGES.items():
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce")
            bad = values.notna() & ((values < low) | (values > high))
            if bad.any():
                out_of_range += int(bad.sum())
                df[col] = values.mask(bad)
    report["out_of_range_values_set_missing"] = out_of_range

    invalid_flags = 0
    for col in BINARY_FEATURES:
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce")
            bad = values.notna() & ~values.isin([0, 1])
            if bad.any():
                invalid_flags += int(bad.sum())
                df[col] = values.mask(bad)
    report["invalid_binary_values_set_missing"] = invalid_flags

    report["missing_values_remaining"] = int(df.isna().sum().sum())
    report["rows_out"] = len(df)
    # Rows were only removed, so the time order from load_data is kept. Re-sorting
    # here would reshuffle rows within each hour and change training results.
    return df.reset_index(drop=True), report


def format_report(report: dict) -> str:
    return "\n".join(f"  {key.replace('_', ' ')}: {value}" for key, value in report.items())
