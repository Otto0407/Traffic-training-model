"""Select the model inputs and drop leaking columns."""
import pandas as pd

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DROPPED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
)


def feature_columns() -> list:
    return CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES


def select_features(df: pd.DataFrame) -> tuple:
    """Return (X, y) with only the allowed feature columns."""
    dropped = {c for cols in DROPPED_FEATURES.values() for c in cols}
    kept = feature_columns()

    overlap = dropped & set(kept)
    if overlap:
        raise ValueError(f"Columns listed as both kept and dropped: {sorted(overlap)}")
    missing = set(kept) - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing from the dataset: {sorted(missing)}")

    unaccounted = set(df.columns) - dropped - set(kept) - {TARGET}
    if unaccounted:
        raise ValueError(
            f"Columns neither kept nor dropped: {sorted(unaccounted)}. "
            "Add them to config.py so every column is a deliberate choice."
        )

    return df[kept].copy(), df[TARGET].copy()
