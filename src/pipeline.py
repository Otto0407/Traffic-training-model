"""Shared split -> preprocess -> train steps, used by main.py and experiments.py."""
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from config import CLASS_ORDER, TARGET
from dataset import make_loader
from features import LEAK_FREE, FeatureSet, select_features
from model import CongestionMLP
from preprocessing import build_preprocessor, encode_labels
from split import random_split, time_split
from train import class_weights, train_model


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_df: pd.DataFrame
    preprocessor: ColumnTransformer
    num_numeric: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(df: pd.DataFrame, cfg, features: FeatureSet = LEAK_FREE, split: str = "time") -> PreparedData:
    """Split the rows, then fit the preprocessor on the training part only."""
    if split == "time":
        train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)
    elif split == "random":
        train_df, val_df, test_df = random_split(df, cfg.train_frac, cfg.val_frac, cfg.seed, TARGET)
    else:
        raise ValueError(f"Unknown split '{split}'")

    X_train, y_train = select_features(train_df, features)
    X_val, y_val = select_features(val_df, features)
    X_test, y_test = select_features(test_df, features)

    preprocessor = build_preprocessor(features)
    return PreparedData(
        X_train=preprocessor.fit_transform(X_train).astype(np.float32),
        y_train=encode_labels(y_train),
        X_val=preprocessor.transform(X_val).astype(np.float32),
        y_val=encode_labels(y_val),
        X_test=preprocessor.transform(X_test).astype(np.float32),
        y_test=encode_labels(y_test),
        test_df=test_df,
        preprocessor=preprocessor,
        num_numeric=len(features.numerical),
    )


def fit_model(data: PreparedData, cfg, device: torch.device, verbose: bool = True) -> tuple:
    """Build and train the MLP described by cfg. Returns (model, history)."""
    set_seed(cfg.seed)
    train_loader = make_loader(
        data.X_train, data.y_train, cfg.batch_size, shuffle=True, seed=cfg.seed, balanced=cfg.balanced_sampling
    )
    val_loader = make_loader(data.X_val, data.y_val, cfg.batch_size, shuffle=False)
    model = CongestionMLP(data.X_train.shape[1], cfg.hidden_sizes, len(CLASS_ORDER), cfg.dropout)
    weights = class_weights(data.y_train, len(CLASS_ORDER)) if cfg.class_weighted_loss else None
    return train_model(model, train_loader, val_loader, cfg, device, weights, data.num_numeric, verbose)
