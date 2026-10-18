"""Wrap NumPy arrays as PyTorch DataLoaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
