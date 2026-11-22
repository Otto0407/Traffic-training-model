import numpy as np

from dataset import make_loader


def test_batches_cover_every_row():
    X, y = np.random.rand(1000, 3).astype(np.float32), np.arange(1000) % 4
    loader = make_loader(X, y, batch_size=128, shuffle=True, seed=0)
    assert sum(len(yb) for _, yb in loader) == 1000


def test_same_seed_gives_same_order():
    X, y = np.random.rand(100, 3).astype(np.float32), np.arange(100) % 4
    first = [yb.tolist() for _, yb in make_loader(X, y, 10, shuffle=True, seed=1)]
    again = [yb.tolist() for _, yb in make_loader(X, y, 10, shuffle=True, seed=1)]
    assert first == again


def test_balanced_sampling_evens_out_classes():
    y = np.array([0] * 900 + [1] * 100)
    X = np.zeros((1000, 1), dtype=np.float32)
    drawn = np.concatenate([yb.numpy() for _, yb in make_loader(X, y, 100, shuffle=True, seed=0, balanced=True)])
    assert 0.4 < drawn.mean() < 0.6  # about half of each class instead of 10%
