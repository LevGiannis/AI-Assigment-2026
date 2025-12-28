#!/bin/bash
set -e
MODE=full
if [[ "$1" == "--quick" ]]; then
  MODE=quick
fi
# Part A: IMDB classic
if [[ "$MODE" == "quick" ]]; then
  python -m src.imdb.vocab --k 100 --n 2 --m 2 --quick
  python -m src.imdb.train_classical --model logreg --config configs/classical_grid.yaml --quick
  python -m src.imdb.train_classical --model bernoulli_nb --config configs/classical_grid.yaml --quick
else
  python -m src.imdb.vocab
  python -m src.imdb.train_classical --model logreg --config configs/classical_grid.yaml
  python -m src.imdb.train_classical --model bernoulli_nb --config configs/classical_grid.yaml
fi
python -m src.imdb.learning_curves ${MODE:+--quick}
python -m src.imdb.eval_test ${MODE:+--quick}
# Part B: IMDB RNN
python -m src.rnn_imdb.train --config configs/imdb_rnn.json ${MODE:+--quick}
python -m src.rnn_imdb.evaluate --config configs/imdb_rnn.json ${MODE:+--quick}
# Part C: FashionMNIST
python -m src.fashion.train --config configs/fashion_cnn.json ${MODE:+--quick}
python -m src.fashion.eval_test --config configs/fashion_cnn.json ${MODE:+--quick}
# Make report
python -m src.make_report
