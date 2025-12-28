Assignment 2 Report
Generated: 2025-12-28 17:54


# Μέρος Α: IMDB Classic


## Vocabulary/IG
![imdb_vocab_ig.png](outputs/plots/imdb_vocab_ig.png)
*Vocabulary IG*

## Learning Curves
![imdb_learning_curves_logreg.png](outputs/plots/imdb_learning_curves_logreg.png)
*Learning curves logreg*

![imdb_learning_curves_bernoulli_nb.png](outputs/plots/imdb_learning_curves_bernoulli_nb.png)
*Learning curves bernoulli_nb*

## Test Results
### logreg
|   class |   precision |   recall |       f1 |   support |
|--------:|------------:|---------:|---------:|----------:|
|       0 |    1        |     0.28 | 0.4375   |        50 |
|       1 |    0.576471 |     1    | 0.731343 |        49 |
### bernoulli_nb
|   class |   precision |   recall |       f1 |   support |
|--------:|------------:|---------:|---------:|----------:|
|       0 |    1        |     0.28 | 0.4375   |        50 |
|       1 |    0.576471 |     1    | 0.731343 |        49 |

# Μέρος Β: IMDB RNN


## Αρχιτεκτονική + Embeddings
![imdb_rnn_architecture.png](outputs/plots/imdb_rnn_architecture.png)
*RNN Architecture*

## Loss Curves
![imdb_rnn_loss_curves.png](outputs/plots/imdb_rnn_loss_curves.png)
*RNN Loss Curves*

## Test Results
|   class |   precision |   recall |       f1 |   support |
|--------:|------------:|---------:|---------:|----------:|
|       0 |         0   |        0 | 0        |        10 |
|       1 |         0.5 |        1 | 0.666667 |        10 |

# Μέρος Γ: FashionMNIST CNN


## CNN Αρχιτεκτονική
![fashion_cnn_architecture.png](outputs/plots/fashion_cnn_architecture.png)
*CNN Architecture*

## Loss Curves
![fashion_loss_curves.png](outputs/plots/fashion_loss_curves.png)
*FashionMNIST Loss Curves*

## Test Results
|   class |   precision |   recall |       f1 |   support |
|--------:|------------:|---------:|---------:|----------:|
|       0 |    0.444444 | 0.5      | 0.470588 |         8 |
|       1 |    1        | 0.846154 | 0.916667 |        13 |
|       2 |    0.48     | 0.857143 | 0.615385 |        14 |
|       3 |    0.8      | 0.444444 | 0.571429 |         9 |
|       4 |    0        | 0        | 0        |        10 |
|       5 |    0.368421 | 0.777778 | 0.5      |         9 |
|       6 |    0.25     | 0.125    | 0.166667 |         8 |
|       7 |    0.642857 | 0.818182 | 0.72     |        11 |
|       8 |    1        | 0.75     | 0.857143 |        12 |
|       9 |    0.75     | 0.5      | 0.6      |         6 |

# Hyperparameters & Selection


See configs/*.json and classical_grid.yaml. Selected by dev set performance.

# Datasets & Splits


IMDB: 20k train/dev/test, FashionMNIST: 60k/10k, see code for split details.

# Αναπαραγωγή


Run:

    bash scripts/run_all.sh --quick
    pytest -q
