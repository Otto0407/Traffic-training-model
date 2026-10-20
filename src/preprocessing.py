"""Feature scaling / encoding, fitted on the training set only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CLASS_ORDER
from features import LEAK_FREE, FeatureSet


def build_preprocessor(features: FeatureSet = LEAK_FREE) -> ColumnTransformer:
    """Fill missing values, then scale / one-hot encode.

    The imputers learn their fill values (median, most frequent) from the
    training set only, like the scaler. Numerical columns come first in the
    output, so input noise can target them by index (see train.py).
    """
    return ColumnTransformer(
        transformers=[
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(features.numerical)),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                list(features.categorical),
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), list(features.binary)),
        ]
    )


def encode_labels(y: pd.Series) -> np.ndarray:
    """Map Low/Moderate/High/Severe to 0..3 in severity order."""
    mapping = {name: i for i, name in enumerate(CLASS_ORDER)}
    return y.map(mapping).to_numpy(dtype=np.int64, copy=True)


def decode_labels(y: np.ndarray) -> list:
    return [CLASS_ORDER[i] for i in y]
