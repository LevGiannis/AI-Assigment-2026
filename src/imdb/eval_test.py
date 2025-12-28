import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from src.imdb.data import load_imdb_data
from src.imdb.preprocess import preprocess_texts
from src.imdb.vectorize import vectorize
from src.imdb.vocab import load_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['logreg', 'bernoulli_nb'])
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--vocab', type=str, default='outputs/vocab.pkl')
    args = parser.parse_args()

    _, test = load_imdb_data(quick=args.quick)
    tokenized_test = preprocess_texts(test['text'])
    vocab = load_vocab(args.vocab)
    X_test, _ = vectorize(tokenized_test, vocab)
    y_test = test['label'].values
    with open(f'outputs/checkpoints/model_{args.model}.pkl', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    metrics = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1])
    micro = precision_recall_fscore_support(y_test, y_pred, average='micro')
    macro = precision_recall_fscore_support(y_test, y_pred, average='macro')
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
    df.to_csv(f'outputs/tables/imdb_test_results_{args.model}.csv', index=False)
    df2.to_csv(f'outputs/tables/imdb_test_results_{args.model}_agg.csv', index=False)
    print(f"Saved test results for {args.model}")

if __name__ == "__main__":
    main()
