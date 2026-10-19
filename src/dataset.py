"""Wrap NumPy arrays as PyTorch DataLoaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0, balanced: bool = False
) -> DataLoader:
    """With balanced=True, rows are drawn with probability inversely proportional
    to their class frequency (random oversampling of the minority classes)."""
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        counts = np.bincount(y)
        row_weights = torch.as_tensor(1.0 / counts[y], dtype=torch.double)
        sampler = WeightedRandomSampler(row_weights, num_samples=len(y), replacement=True, generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
