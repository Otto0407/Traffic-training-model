from dataclasses import replace

import numpy as np
import pytest
import torch

from config import TrainConfig
from dataset import make_loader
from model import CongestionMLP
from pipeline import fit_model
from train import class_weights, train_model


def test_class_weights_are_inverse_frequency_with_mean_one():
    y = np.array([0] * 60 + [1] * 20 + [2] * 10 + [3] * 10)
    w = class_weights(y, 4).numpy()
    frequency = np.bincount(y) / len(y)
    assert (w * frequency).sum() == pytest.approx(1.0)  # mean weight over training rows
    assert (w * np.bincount(y)) == pytest.approx(np.full(4, 25.0))  # each class contributes equally
    assert w[0] < w[1] < w[2] == pytest.approx(w[3])


def _loaders(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    return make_loader(X, y, 64, shuffle=True, seed=seed), make_loader(X, y, 64, shuffle=False)


def test_early_stopping_stops_after_patience_epochs(cpu):
    # With a learning rate of 0 the validation loss never improves after epoch 1.
    cfg = replace(TrainConfig(), learning_rate=0.0, max_epochs=50, patience=3)
    model, history = train_model(CongestionMLP(6, [8], 4, 0.0), *_loaders(), cfg, cpu, verbose=False)
    assert history["best_epoch"] == 1
    assert len(history["train_loss"]) == 4


def test_best_weights_are_restored(cpu):
    cfg = replace(TrainConfig(), max_epochs=8, patience=100)
    train, val = _loaders()
    model, history = train_model(CongestionMLP(6, [16], 4, 0.0), train, val, cfg, cpu, verbose=False)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        loss = sum(criterion(model(xb), yb).item() * len(yb) for xb, yb in val) / 400
    assert loss == pytest.approx(min(history["val_loss"]), rel=1e-5)


def test_without_restore_the_last_weights_are_kept(cpu):
    cfg = replace(TrainConfig(), max_epochs=4, patience=100, restore_best_weights=False)
    train, val = _loaders()
    _, history = train_model(CongestionMLP(6, [16], 4, 0.0), train, val, cfg, cpu, verbose=False)
    assert len(history["val_loss"]) == 4


@pytest.mark.parametrize("overrides", [
    {},
    {"class_weighted_loss": True},
    {"balanced_sampling": True},
    {"input_noise": 0.1},
    {"optimizer": "adamw", "weight_decay": 1e-3},
])
@pytest.mark.slow
def test_training_options_run_and_learn(prepared, cpu, overrides):
    cfg = replace(TrainConfig(), max_epochs=2, hidden_sizes=[32], **overrides)
    model, history = fit_model(prepared, cfg, cpu, verbose=False)
    assert len(history["train_loss"]) == 2
    assert history["train_loss"][1] < history["train_loss"][0]  # it is learning


@pytest.mark.slow
def test_same_seed_gives_identical_training(prepared, cpu):
    cfg = replace(TrainConfig(), max_epochs=1, hidden_sizes=[16])
    _, a = fit_model(prepared, cfg, cpu, verbose=False)
    _, b = fit_model(prepared, cfg, cpu, verbose=False)
    assert a["val_loss"] == b["val_loss"]
