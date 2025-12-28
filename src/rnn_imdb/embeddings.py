import numpy as np
import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    embeddings = np.random.normal(0, 1, (len(vocab), embedding_dim)).astype(np.float32)
    word2idx = {w: i for i, w in enumerate(vocab)}
    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            word = parts[0]
            if word in word2idx:
                embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    # <pad> = zeros, <unk> = random
    if PAD_TOKEN in word2idx:
        embeddings[word2idx[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)
    if UNK_TOKEN in word2idx:
        embeddings[word2idx[UNK_TOKEN]] = np.random.normal(0, 1, embedding_dim).astype(np.float32)
    return torch.tensor(embeddings), found
