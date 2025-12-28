import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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

    train_df, test_df = load_imdb_data(quick=args.quick)
    train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])
    # Build vocab
    all_tokens = [t for doc in train_df['text'] for t in doc]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for t in set(all_tokens):
        if t not in vocab:
            vocab[t] = len(vocab)
    # Embeddings
    emb_matrix, found = load_glove_embeddings(args.glove_path, vocab, config['embedding_dim'])
    print(f"Loaded {found} pretrained vectors.")
    # Datasets
    train_ds = IMDBDataset(train_df['text'].tolist(), train_df['label'].tolist(), vocab, max_len=config['max_len'])
    dev_ds = IMDBDataset(dev_df['text'].tolist(), dev_df['label'].tolist(), vocab, max_len=config['max_len'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=config['batch_size'])
    # Model
    model = RNNIMDB(len(vocab), config['embedding_dim'], config['hidden_dim'], config['num_layers'], embeddings=emb_matrix, bidirectional=True, dropout=config['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # Training
    best_dev = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'dev_loss': [], 'best_epoch': 0}
    num_epochs = 2 if args.quick else config['epochs']
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                dev_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        avg_dev = np.mean(dev_losses)
        history['train_loss'].append(avg_train)
        history['dev_loss'].append(avg_dev)
        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f} dev_loss={avg_dev:.4f}")
        if avg_dev < best_dev:
            best_dev = avg_dev
            best_epoch = epoch
            os.makedirs('outputs/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'outputs/checkpoints/imdb_rnn_best.pt')
            history['best_epoch'] = best_epoch
            print(f"New best epoch: {best_epoch+1}")
    # Save loss curves
    os.makedirs('outputs/plots', exist_ok=True)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['dev_loss'], label='dev')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('outputs/plots/imdb_rnn_loss_curves.png')
    plt.close()
    # Save history
    os.makedirs('outputs/tables', exist_ok=True)
    pd.DataFrame(history).to_csv('outputs/tables/imdb_rnn_history.csv', index=False)
    print(f"Best epoch: {best_epoch+1}")

if __name__ == "__main__":
    main()
