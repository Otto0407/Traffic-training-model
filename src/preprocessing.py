"""Feature scaling / encoding, fitted on the training set only."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BINARY_FEATURES, CATEGORICAL_FEATURES, CLASS_ORDER, NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def encode_labels(y: pd.Series) -> np.ndarray:
    """Map Low/Moderate/High/Severe to 0..3 in severity order."""
    mapping = {name: i for i, name in enumerate(CLASS_ORDER)}
    return y.map(mapping).to_numpy(dtype=np.int64, copy=True)


def decode_labels(y: np.ndarray) -> list:
    return [CLASS_ORDER[i] for i in y]
