import numpy as np
import pandas as pd

from baseline import lookup_baseline, majority_baseline
from config import TARGET


def test_majority_baseline_predicts_most_common_class():
    assert (majority_baseline(np.array([0, 0, 0, 3, 1]), 4) == 0).all()


def test_lookup_baseline_uses_most_common_label_per_key():
    train = pd.DataFrame({"hour": [8, 8, 8, 2, 2], TARGET: ["Severe", "Severe", "High", "Low", "Low"]})
    test = pd.DataFrame({"hour": [8, 2, 13]})  # 13 never seen
    preds = lookup_baseline(train, test, ["hour"])
    assert list(preds[:2]) == [3, 0]
    # Unseen key falls back to the overall most common label (Low and Severe tie at 2;
    # mode() returns the first in sort order, "Low").
    assert preds[2] == 0


def test_lookup_baseline_with_several_keys(clean_df):
    train, test = clean_df.iloc[:20000], clean_df.iloc[20000:21000]
    preds = lookup_baseline(train, test, ["hour", "is_weekend", "city_zone"])
    assert len(preds) == len(test)
