import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
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

    train, _ = load_imdb_data(quick=args.quick)
    train, dev = train_test_split(train, test_size=0.2, random_state=42, stratify=train['label'])
    tokenized_train = preprocess_texts(train['text'])
    tokenized_dev = preprocess_texts(dev['text'])
    vocab = load_vocab(args.vocab)
    X_train, _ = vectorize(tokenized_train, vocab)
    X_dev, _ = vectorize(tokenized_dev, vocab)
    y_train = train['label'].values
    y_dev = dev['label'].values

    if args.model == 'logreg':
        model = LogisticRegression(solver='liblinear')
    else:
        model = BernoulliNB()
    # Learning curve: train on increasing #examples
    sizes = np.linspace(10, X_train.shape[0], 5, dtype=int) if not args.quick else [10, 50, X_train.shape[0]]
    results = []
    for n in sizes:
        model.fit(X_train[:n], y_train[:n])
        y_pred_train = model.predict(X_train[:n])
        y_pred_dev = model.predict(X_dev)
        prf_train = precision_recall_fscore_support(y_train[:n], y_pred_train, average=None, labels=[0,1])
        prf_dev = precision_recall_fscore_support(y_dev, y_pred_dev, average=None, labels=[0,1])
        results.append({
            'n': n,
            'train_precision_pos': prf_train[0][1],
            'train_recall_pos': prf_train[1][1],
            'train_f1_pos': prf_train[2][1],
            'dev_precision_pos': prf_dev[0][1],
            'dev_recall_pos': prf_dev[1][1],
            'dev_f1_pos': prf_dev[2][1],
        })
    df = pd.DataFrame(results)
    os.makedirs('outputs/tables', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    df.to_csv(f'outputs/tables/imdb_learning_curve_{args.model}.csv', index=False)
    # Plot
    plt.figure()
    plt.plot(df['n'], df['train_f1_pos'], label='Train F1 (pos)')
    plt.plot(df['n'], df['dev_f1_pos'], label='Dev F1 (pos)')
    plt.xlabel('#Train examples')
    plt.ylabel('F1 (positive)')
    plt.title(f'Learning Curve ({args.model})')
    plt.legend()
    plt.savefig(f'outputs/plots/imdb_learning_curves_{args.model}.png')
    plt.close()
    print(f"Saved learning curve for {args.model}")

if __name__ == "__main__":
    main()
