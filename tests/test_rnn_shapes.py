import torch
from src.rnn_imdb.model import RNNIMDB

def test_forward_shape():
    model = RNNIMDB(vocab_size=10, embedding_dim=8, hidden_dim=4, num_layers=2)
    x = torch.randint(0, 10, (5, 7))  # (B, T)
    logits = model(x)
    assert logits.shape == (5, 2)

def test_global_max_pool():
    model = RNNIMDB(vocab_size=10, embedding_dim=8, hidden_dim=4, num_layers=2)
    x = torch.randint(0, 10, (3, 6))
    out = model.embedding(x)
    rnn_out, _ = model.rnn(out)
    pooled, _ = torch.max(rnn_out, dim=1)
    assert pooled.shape[0] == 3 and pooled.shape[1] == (4*2)
