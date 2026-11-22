"""Shared fixtures. Data fixtures are session-scoped and must not be modified
in place by tests: take a .copy() first."""
import pytest
import torch

from cleaning import clean_data
from config import DEFAULT_DATA_PATH, TrainConfig
from data_loader import load_data
from features import select_features
from pipeline import prepare_data
from preprocessing import build_preprocessor

# Keep the test run light so it can run next to training jobs.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def raw_data():
    return load_data(DEFAULT_DATA_PATH)


@pytest.fixture(scope="session")
def clean_df(raw_data):
    return clean_data(raw_data)[0]


@pytest.fixture(scope="session")
def small_df(clean_df):
    """About 10 intersections over the full period. Every hour is present, so
    the time split behaves as on the full data, and the intersections are
    chosen so every zone and road type appears (the encoded input keeps all
    72 columns), at about a tenth of the size."""
    sites = clean_df.drop_duplicates("intersection_id").sort_values("intersection_id")
    ids = []
    for col in ("city_zone", "road_type"):
        for value in sites[col].unique():
            if value not in set(sites[sites["intersection_id"].isin(ids)][col]):
                ids.append(sites[sites[col] == value]["intersection_id"].iloc[0])
    ids += [i for i in sites["intersection_id"] if i not in ids][: max(0, 10 - len(ids))]
    return clean_df[clean_df["intersection_id"].isin(ids)].reset_index(drop=True)


@pytest.fixture(scope="session")
def prepared(small_df):
    return prepare_data(small_df, TrainConfig())


@pytest.fixture(scope="session")
def fitted_preprocessor(small_df):
    return build_preprocessor().fit(select_features(small_df)[0])


@pytest.fixture
def cpu():
    return torch.device("cpu")
