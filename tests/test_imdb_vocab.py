import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
from src.imdb import vocab, vectorize

def test_vocab_size_and_removal():
    # Synthetic data
    texts = [["a", "b", "c"], ["a", "b"], ["a", "d"], ["e"]]
    labels = [0, 1, 0, 1]
    df = vocab.compute_df(texts)
    most, least = vocab.remove_top_bottom(df, n=1, k=1)
    filtered = [w for w in df if w not in most and w not in least]
    igs = vocab.compute_ig(texts, labels, filtered)
    top = vocab.select_top_m_ig(igs, m=2)
    # Save and load vocab
    vocab.save_vocab(top, "tmp_vocab.pkl")
    loaded = vocab.load_vocab("tmp_vocab.pkl")
    os.remove("tmp_vocab.pkl")
    assert len(top) == 2
    assert set(top).isdisjoint(most)
    assert set(top).isdisjoint(least)
    assert loaded == top

def test_ig_ranking():
    # Known IG: word 'a' appears in all docs, so IG=0
    texts = [["a", "b"], ["a", "c"], ["a", "d"]]
    labels = [0, 1, 0]
    df = vocab.compute_df(texts)
    igs = vocab.compute_ig(texts, labels, df.keys())
    assert igs['a'] < igs['b'] and igs['a'] < igs['c'] and igs['a'] < igs['d']
