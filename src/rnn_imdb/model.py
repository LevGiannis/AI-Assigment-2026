import torch
import torch.nn as nn

class RNNIMDB(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embeddings=None, bidirectional=True, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        # Global max pooling
        pooled, _ = torch.max(out, dim=1)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
