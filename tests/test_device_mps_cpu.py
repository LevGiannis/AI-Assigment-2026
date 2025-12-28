import torch
from src.rnn_imdb.train import get_device

def test_device_cpu():
    d = get_device()
    assert str(d) in ["cpu", "cuda", "mps"]
    # Simulate no mps
    orig = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    d2 = get_device()
    assert str(d2) in ["cpu", "cuda"]
    torch.backends.mps.is_available = orig
