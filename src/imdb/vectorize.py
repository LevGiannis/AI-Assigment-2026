import numpy as np
import scipy.sparse as sp

def vectorize(tokenized_texts, vocab):
    word2idx = {w: i for i, w in enumerate(vocab)}
    rows, cols, data = [], [], []
    for row, doc in enumerate(tokenized_texts):
        for w in set(doc):
            if w in word2idx:
                rows.append(row)
                cols.append(word2idx[w])
                data.append(1)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(tokenized_texts), len(vocab)), dtype=np.int8)
    return mat, word2idx
