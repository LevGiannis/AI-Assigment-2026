import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        ids = [self.vocab.get(t, self.vocab[UNK_TOKEN]) for t in tokens]
        ids = ids[:self.max_len]
        ids += [self.vocab[PAD_TOKEN]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

def load_imdb_data(quick=False):
    # For quick: tiny synthetic
    if quick:
        data = [(["good", "movie"], 1), (["bad", "movie"], 0), (["excellent"], 1), (["awful"], 0)] * 10
        df = pd.DataFrame(data, columns=["text", "label"])
        train, test = train_test_split(df, test_size=0.5, random_state=42, stratify=df["label"])
        return train.reset_index(drop=True), test.reset_index(drop=True)
    else:
        # Placeholder for real IMDB
        raise NotImplementedError("Real IMDB loading not implemented yet.")
