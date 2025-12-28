import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from src.imdb.data import load_imdb_data
from src.imdb.preprocess import preprocess_texts
from src.imdb.vectorize import vectorize
from src.imdb.vocab import load_vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['logreg', 'bernoulli_nb'])
    parser.add_argument('--config', type=str, default=None)
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
        param_grid = {'C': [0.1, 1, 10]} if not args.quick else {'C': [1]}
    else:
        model = BernoulliNB()
        param_grid = {'alpha': [0.1, 1, 10]} if not args.quick else {'alpha': [1]}

    gs = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_dev)
    metrics = precision_recall_fscore_support(y_dev, y_pred, average=None, labels=[0,1])
    micro = precision_recall_fscore_support(y_dev, y_pred, average='micro')
    macro = precision_recall_fscore_support(y_dev, y_pred, average='macro')
    os.makedirs('outputs/tables', exist_ok=True)
    df = pd.DataFrame({
        'class': [0,1],
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2],
        'support': metrics[3]
    })
    df.to_csv(f'outputs/tables/imdb_dev_gridsearch.csv', index=False)
    with open(f'outputs/checkpoints/model_{args.model}.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best params: {gs.best_params_}")
    print(f"Saved model and dev metrics for {args.model}")

if __name__ == "__main__":
    main()
