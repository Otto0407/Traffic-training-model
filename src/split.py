"""Chronological train/validation/test split."""
import pandas as pd

from config import TIMESTAMP


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple:
    """Split by time so that validation and test lie strictly in the future.

    The cut points are chosen on the unique timestamps, so every intersection
    observed at a given hour lands in the same subset and no hour is shared
    between subsets.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Need 0 < train_frac, val_frac and train_frac + val_frac < 1")

    times = df[TIMESTAMP].drop_duplicates().sort_values().reset_index(drop=True)
    train_end = times.iloc[int(len(times) * train_frac)]
    val_end = times.iloc[int(len(times) * (train_frac + val_frac))]

    train = df[df[TIMESTAMP] < train_end]
    val = df[(df[TIMESTAMP] >= train_end) & (df[TIMESTAMP] < val_end)]
    test = df[df[TIMESTAMP] >= val_end]
    return train, val, test
