import pickle
import hashlib
import os
from src.imdb.vocab import save_vocab, load_vocab
from src.imdb.train_classical import main as train_main
import sys
import shutil

def test_common_vocab(tmp_path, monkeypatch):
    # Create a dummy vocab
    vocab = ["a", "b", "c"]
    vocab_path = tmp_path / "vocab.pkl"
    save_vocab(vocab, str(vocab_path))
    # Hash vocab
    with open(vocab_path, "rb") as f:
        vocab_bytes = f.read()
    vocab_hash = hashlib.md5(vocab_bytes).hexdigest()
    # Simulate both models using same vocab
    # (Here, just check that both load the same file)
    v1 = load_vocab(str(vocab_path))
    v2 = load_vocab(str(vocab_path))
    assert v1 == v2
    # Hash check
    with open(vocab_path, "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == vocab_hash
