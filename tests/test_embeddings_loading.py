import tempfile
import numpy as np
from src.rnn_imdb.embeddings import load_glove_embeddings, PAD_TOKEN, UNK_TOKEN

def test_embedding_loading():
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "foo": 2, "bar": 3}
    # Fake glove file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("foo 0.1 0.2 0.3\n")
        f.write("bar 0.4 0.5 0.6\n")
        path = f.name
    emb, found = load_glove_embeddings(path, vocab, embedding_dim=3)
    assert emb.shape == (4, 3)
    assert np.allclose(emb[vocab["foo"]].numpy(), [0.1, 0.2, 0.3])
    assert np.allclose(emb[vocab["bar"]].numpy(), [0.4, 0.5, 0.6])
    assert np.allclose(emb[vocab[PAD_TOKEN]].numpy(), np.zeros(3))
    # <unk> is not zeros
    assert not np.allclose(emb[vocab[UNK_TOKEN]].numpy(), np.zeros(3))
