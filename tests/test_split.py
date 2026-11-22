import pytest

from config import TARGET, TIMESTAMP
from split import random_split, time_split


def test_time_split_is_strictly_chronological(small_df):
    train, val, test = time_split(small_df, 0.6, 0.2)
    assert train[TIMESTAMP].max() < val[TIMESTAMP].min()
    assert val[TIMESTAMP].max() < test[TIMESTAMP].min()


def test_time_split_keeps_each_hour_in_one_subset(small_df):
    train, val, test = time_split(small_df, 0.6, 0.2)
    hours = [set(part[TIMESTAMP]) for part in (train, val, test)]
    assert not (hours[0] & hours[1]) and not (hours[1] & hours[2]) and not (hours[0] & hours[2])


def test_time_split_covers_every_row_with_expected_shares(small_df):
    train, val, test = time_split(small_df, 0.6, 0.2)
    assert len(train) + len(val) + len(test) == len(small_df)
    assert len(train) / len(small_df) == pytest.approx(0.6, abs=0.01)
    assert len(val) / len(small_df) == pytest.approx(0.2, abs=0.01)


@pytest.mark.parametrize("train_frac,val_frac", [(0, 0.2), (0.6, 0), (0.8, 0.2), (1.2, 0.1)])
def test_time_split_rejects_bad_fractions(small_df, train_frac, val_frac):
    with pytest.raises(ValueError):
        time_split(small_df, train_frac, val_frac)


def test_random_split_is_stratified(small_df):
    train, val, test = random_split(small_df, 0.6, 0.2, seed=0, stratify_col=TARGET)
    overall = small_df[TARGET].value_counts(normalize=True)
    for part in (train, val, test):
        assert (part[TARGET].value_counts(normalize=True) - overall).abs().max() < 0.01
    assert len(train) + len(val) + len(test) == len(small_df)
