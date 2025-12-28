import numpy as np
import scipy.sparse as sp
from src.imdb.vectorize import vectorize

def test_sparse_and_binary():
    texts = [["a", "b"], ["b", "c"], ["a"]]
    vocab = ["a", "b", "c"]
    mat, word2idx = vectorize(texts, vocab)
    assert sp.issparse(mat)
    arr = mat.toarray()
    assert np.all((arr == 0) | (arr == 1))

def test_vocab_index_mapping():
    texts = [["a", "b"], ["b", "c"]]
    vocab = ["a", "b", "c"]
    _, word2idx1 = vectorize(texts, vocab)
    _, word2idx2 = vectorize(texts, vocab)
    assert word2idx1 == word2idx2
