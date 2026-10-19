"""Non-learned baselines the MLP must beat."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from config import TARGET
from preprocessing import encode_labels


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the most frequent training class for every test sample."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf.predict(np.zeros((n_test, 1)))


def lookup_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, keys: list) -> np.ndarray:
    """Predict the most frequent training label for each combination of `keys`
    (e.g. hour of day). A non-learned reference: if the MLP cannot beat a
    lookup table, it has not learned anything beyond those keys."""
    fallback = train_df[TARGET].mode()[0]
    table = train_df.groupby(keys)[TARGET].agg(lambda s: s.mode()[0])
    labels = test_df.set_index(keys).index.map(lambda k: table.get(k, fallback))
    return encode_labels(pd.Series(list(labels)))
