import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from src.rnn_imdb.data import load_imdb_data, IMDBDataset, set_seed, PAD_TOKEN, UNK_TOKEN
from src.rnn_imdb.embeddings import load_glove_embeddings
from src.rnn_imdb.model import RNNIMDB

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--glove_path', type=str, required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    set_seed(config.get('seed', 42))
    device = get_device()
    print(f"Using device: {device}")

    _, test_df = load_imdb_data(quick=args.quick)
    # Build vocab
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for t in set([t for doc in test_df['text'] for t in doc]):
        if t not in vocab:
            vocab[t] = len(vocab)
    emb_matrix, _ = load_glove_embeddings(args.glove_path, vocab, config['embedding_dim'])
    test_ds = IMDBDataset(test_df['text'].tolist(), test_df['label'].tolist(), vocab, max_len=config['max_len'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'])
    model = RNNIMDB(len(vocab), config['embedding_dim'], config['hidden_dim'], config['num_layers'], embeddings=emb_matrix, bidirectional=True, dropout=config['dropout']).to(device)
    model.load_state_dict(torch.load('outputs/checkpoints/imdb_rnn_best.pt', map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())
    metrics = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0,1])
    micro = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    macro = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    os.makedirs('outputs/tables', exist_ok=True)
    df = pd.DataFrame({
        'class': [0,1],
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2],
        'support': metrics[3]
    })
    df2 = pd.DataFrame({
        'type': ['micro', 'macro'],
        'precision': [micro[0], macro[0]],
        'recall': [micro[1], macro[1]],
        'f1': [micro[2], macro[2]],
        'support': [micro[3], macro[3]]
    })
    df.to_csv('outputs/tables/imdb_rnn_test_results.csv', index=False)
    df2.to_csv('outputs/tables/imdb_rnn_test_results_agg.csv', index=False)
    print("Saved test results.")

if __name__ == "__main__":
    main()
