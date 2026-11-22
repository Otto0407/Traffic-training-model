import numpy as np
import pandas as pd

from config import CLASS_ORDER, NUMERICAL_FEATURES
from features import FeatureSet, select_features
from preprocessing import build_preprocessor, decode_labels, encode_labels


def test_output_has_44_columns(prepared):
    # one-hot: hour 24 + day_of_week 7 + city_zone 6 + weather 4, plus 3 flags
    assert prepared.X_train.shape[1] == 44
    assert prepared.num_numeric == len(NUMERICAL_FEATURES) == 0


def test_encoders_are_fitted_on_training_rows_only(prepared):
    encoder = prepared.preprocessor.named_transformers_["cat"][-1]
    train_X, _ = select_features(prepared.train_df)
    for col, cats in zip(train_X.columns, encoder.categories_):
        assert set(cats) == set(train_X[col].unique())


def test_scaler_and_median_fill_use_training_data(small_df):
    # The final model has no numerical inputs, but the numerical branch must
    # still work (and learn from the training rows only) if one is added.
    fs = FeatureSet(("hour",), ("temperature",), ("public_event",))
    train = small_df.head(2000)
    pre = build_preprocessor(fs).fit(train[fs.columns])
    imputer, scaler = pre.named_transformers_["num"]
    assert imputer.statistics_[0] == train["temperature"].median()
    assert np.isclose(scaler.mean_[0], train["temperature"].mean())
    gappy = train[fs.columns].head(2).copy()
    gappy.loc[gappy.index[0], "temperature"] = np.nan
    assert not np.isnan(pre.transform(gappy).astype(float)).any()


def test_missing_values_are_filled_from_training_data(small_df):
    X, _ = select_features(small_df)
    pre = build_preprocessor().fit(X)
    gappy = X.head(2).copy()
    gappy.loc[gappy.index[0], "weather_condition"] = np.nan
    gappy.loc[gappy.index[1], "public_event"] = np.nan
    out = pre.transform(gappy).astype(float)
    assert not np.isnan(out).any()
    assert pre.named_transformers_["bin"].statistics_[1] == X["public_event"].mode()[0]


def test_unseen_category_does_not_crash(fitted_preprocessor, small_df):
    X, _ = select_features(small_df.head(1))
    X = X.assign(weather_condition="Sandstorm")
    assert fitted_preprocessor.transform(X).shape == (1, 44)


def test_labels_round_trip_in_severity_order():
    labels = pd.Series(["Severe", "Low", "High", "Moderate"])
    codes = encode_labels(labels)
    assert list(codes) == [3, 0, 2, 1]
    assert decode_labels(codes) == list(labels)
    assert CLASS_ORDER == ["Low", "Moderate", "High", "Severe"]
