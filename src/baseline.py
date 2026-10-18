"""Majority-class baseline: the minimum performance the MLP must beat."""
import numpy as np
from sklearn.dummy import DummyClassifier


def majority_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the most frequent training class for every test sample."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    return clf.predict(np.zeros((n_test, 1)))
