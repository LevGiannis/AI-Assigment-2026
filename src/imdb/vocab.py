import collections
import numpy as np
import pickle
import os
from .preprocess import preprocess_texts

def compute_df(tokenized_texts):
    df = collections.Counter()
    for doc in tokenized_texts:
        df.update(set(doc))
    return df

def remove_top_bottom(df, n, k):
    # Remove top-n frequent and bottom-k rare words
    most_common = set([w for w, _ in df.most_common(n)])
    least_common = set([w for w, _ in df.most_common()[-k:]])
    return most_common, least_common

def compute_ig(tokenized_texts, labels, vocab):
    # Binary IG for each word in vocab
    N = len(tokenized_texts)
    label_set = sorted(set(labels))
    label_counts = {l: sum(1 for y in labels if y==l) for l in label_set}
    H_C = -sum((label_counts[l]/N)*np.log2(label_counts[l]/N) for l in label_set if label_counts[l]>0)
    igs = {}
    for w in vocab:
        present = np.array([w in doc for doc in tokenized_texts])
        ig = H_C
        for v in [0,1]:
            idx = (present==v)
            if np.sum(idx)==0:
                continue
            sub_labels = np.array(labels)[idx]
            sub_counts = [np.sum(sub_labels==l) for l in label_set]
            sub_N = np.sum(idx)
            H = -sum((c/sub_N)*np.log2(c/sub_N) for c in sub_counts if c>0)
            ig -= (sub_N/N)*H
        igs[w] = ig
    return igs

def select_top_m_ig(igs, m):
    return [w for w, _ in sorted(igs.items(), key=lambda x: -x[1])[:m]]

def save_vocab(vocab, path):
    with open(path, "wb") as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# CLI entrypoint
if __name__ == "__main__":
    import argparse
    from src.imdb.data import load_imdb_data
    from src.imdb.preprocess import preprocess_texts
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, help='Remove bottom-k rare words')
    parser.add_argument('--n', type=int, required=True, help='Remove top-n frequent words')
    parser.add_argument('--m', type=int, required=True, help='Select top-m IG words')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--out', type=str, default='outputs/vocab.pkl')
    args = parser.parse_args()

    train, _ = load_imdb_data(quick=args.quick)
    tokenized = preprocess_texts(train['text'])
    df = compute_df(tokenized)
    most_common, least_common = remove_top_bottom(df, args.n, args.k)
    filtered_vocab = [w for w in df if w not in most_common and w not in least_common]
    igs = compute_ig(tokenized, train['label'].tolist(), filtered_vocab)
    vocab = select_top_m_ig(igs, args.m)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_vocab(vocab, args.out)
    print(f"Saved vocab of size {len(vocab)} to {args.out}")
    print(f"Removed top-n: {args.n}, bottom-k: {args.k}, selected top-m IG: {args.m}")
