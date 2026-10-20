"""Select the model inputs and drop leaking columns."""
from dataclasses import dataclass

import pandas as pd

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DROPPED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    TIMESTAMP,
)


@dataclass(frozen=True)
class FeatureSet:
    categorical: tuple
    numerical: tuple
    binary: tuple

    @property
    def columns(self) -> list:
        return list(self.categorical) + list(self.numerical) + list(self.binary)

    def without(self, *cols) -> "FeatureSet":
        drop = set(cols)
        return FeatureSet(
            tuple(c for c in self.categorical if c not in drop),
            tuple(c for c in self.numerical if c not in drop),
            tuple(c for c in self.binary if c not in drop),
        )


# The feature set used by the final model (see config.py for the reasoning).
LEAK_FREE = FeatureSet(tuple(CATEGORICAL_FEATURES), tuple(NUMERICAL_FEATURES), tuple(BINARY_FEATURES))


def all_columns_feature_set(df: pd.DataFrame) -> FeatureSet:
    """Every column except the record id, timestamp and target, as the
    proposal originally planned: numeric columns scaled, text columns one-hot."""
    cols = [c for c in df.columns if c not in ("record_id", TIMESTAMP, TARGET)]
    numeric = tuple(c for c in cols if pd.api.types.is_numeric_dtype(df[c]))
    categorical = tuple(c for c in cols if c not in numeric)
    return FeatureSet(categorical, numeric, ())


def feature_columns() -> list:
    return LEAK_FREE.columns


def select_features(df: pd.DataFrame, features: FeatureSet = LEAK_FREE) -> tuple:
    """Return (X, y) with only the allowed feature columns."""
    kept = features.columns
    missing = set(kept) - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing from the dataset: {sorted(missing)}")

    if features == LEAK_FREE:
        dropped = {c for cols in DROPPED_FEATURES.values() for c in cols}
        overlap = dropped & set(kept)
        if overlap:
            raise ValueError(f"Columns listed as both kept and dropped: {sorted(overlap)}")
        unaccounted = set(df.columns) - dropped - set(kept) - {TARGET}
        if unaccounted:
            raise ValueError(
                f"Columns neither kept nor dropped: {sorted(unaccounted)}. "
                "Add them to config.py so every column is a deliberate choice."
            )

    return df[kept].copy(), df[TARGET].copy()


VEHICLE_RATIOS = ("heavy_vehicle_ratio", "motorcycle_ratio", "public_transport_ratio", "other_vehicle_ratio")


def add_vehicle_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Vehicle mix as shares of vehicle_count. Dividing by the total removes
    the traffic-volume leak; only used in experiments.py (it did not help)."""
    out = df.copy()
    total = out["vehicle_count"]
    other = total - out["heavy_vehicle_count"] - out["motorcycle_count"] - out["public_transport_count"]
    out["heavy_vehicle_ratio"] = out["heavy_vehicle_count"] / total
    out["motorcycle_ratio"] = out["motorcycle_count"] / total
    out["public_transport_ratio"] = out["public_transport_count"] / total
    out["other_vehicle_ratio"] = other / total
    return out
