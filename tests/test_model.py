import torch

from model import CongestionMLP


def test_output_shape_and_parameter_count():
    model = CongestionMLP(72, [256, 128, 64], 4, 0.3)
    assert model(torch.zeros(5, 72)).shape == (5, 4)
    assert sum(p.numel() for p in model.parameters()) == 60_100


def test_dropout_only_active_in_training_mode():
    torch.manual_seed(0)
    model = CongestionMLP(10, [32], 4, 0.5)
    x = torch.randn(8, 10)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))
