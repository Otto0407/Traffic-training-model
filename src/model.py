"""Multi-Layer Perceptron for congestion level classification."""
import torch
from torch import nn


class CongestionMLP(nn.Module):
    """Input -> [Linear -> ReLU -> Dropout] x len(hidden_sizes) -> Linear(num_classes).

    The output is raw logits; CrossEntropyLoss applies softmax internally.
    """

    def __init__(self, input_dim: int, hidden_sizes: list, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(dropout)]
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
