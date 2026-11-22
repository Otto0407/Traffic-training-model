import numpy as np
import pandas as pd

from config import CLASS_ORDER, NUMERICAL_FEATURES
from features import LEAK_FREE, select_features
from preprocessing import build_preprocessor, decode_labels, encode_labels


def test_output_has_72_columns_with_numericals_first(prepared):
    assert prepared.X_train.shape[1] == 72
    assert prepared.num_numeric == len(NUMERICAL_FEATURES)
    # Standardised numericals: mean ~0, std ~1 on the training set.
    num = prepared.X_train[:, : prepared.num_numeric]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-5)


def test_scaler_is_fitted_on_training_rows_only(prepared):
    scaler = prepared.preprocessor.named_transformers_["num"][-1]
    train_X, _ = select_features(prepared.train_df)
    assert np.allclose(scaler.mean_, train_X[list(NUMERICAL_FEATURES)].mean().to_numpy())


def test_missing_values_are_filled_from_training_data(small_df):
    X, _ = select_features(small_df)
    pre = build_preprocessor().fit(X)
    gappy = X.head(3).copy()
    gappy.loc[gappy.index[0], "temperature"] = np.nan
    gappy.loc[gappy.index[1], "weather_condition"] = np.nan
    gappy.loc[gappy.index[2], "is_holiday"] = np.nan
    out = pre.transform(gappy).astype(float)
    assert not np.isnan(out).any()
    median = pre.named_transformers_["num"][0].statistics_[NUMERICAL_FEATURES.index("temperature")]
    assert median == X["temperature"].median()


def test_unseen_category_does_not_crash(fitted_preprocessor, small_df):
    X, _ = select_features(small_df.head(1))
    X = X.assign(weather_condition="Sandstorm")
    assert fitted_preprocessor.transform(X).shape == (1, 72)


def test_labels_round_trip_in_severity_order():
    labels = pd.Series(["Severe", "Low", "High", "Moderate"])
    codes = encode_labels(labels)
    assert list(codes) == [3, 0, 2, 1]
    assert decode_labels(codes) == list(labels)
    assert CLASS_ORDER == ["Low", "Moderate", "High", "Severe"]
