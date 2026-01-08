#!/bin/bash
set -e
MODE=full
QUICK_FLAG=""
PYTHON=${PYTHON:-python}
if [[ "$1" == "--quick" ]]; then
  MODE=quick
  QUICK_FLAG="--quick"
fi
# Part A: IMDB classic
if [[ "$MODE" == "quick" ]]; then
  $PYTHON -m src.imdb.vocab --k 100 --n 2 --m 2 --quick
  $PYTHON -m src.imdb.train_classical --model logreg --config configs/classical_grid.yaml --quick
  $PYTHON -m src.imdb.train_classical --model bernoulli_nb --config configs/classical_grid.yaml --quick
else
  # Full mode needs explicit vocab parameters (k/n/m are required by vocab CLI)
  $PYTHON -m src.imdb.vocab --k 15 --n 100 --m 5000
  $PYTHON -m src.imdb.train_classical --model logreg --config configs/classical_grid.yaml
  $PYTHON -m src.imdb.train_classical --model bernoulli_nb --config configs/classical_grid.yaml
fi
for MODEL in logreg bernoulli_nb; do
  $PYTHON -m src.imdb.learning_curves --model $MODEL $QUICK_FLAG
  $PYTHON -m src.imdb.eval_test --model $MODEL $QUICK_FLAG
done
# Part B: IMDB RNN
GLOVE_PATH=${GLOVE_PATH:-""}
if [[ "$MODE" == "quick" ]]; then
  if [[ -z "$GLOVE_PATH" ]]; then
    mkdir -p outputs
    TINY_GLOVE="outputs/glove_tiny_100d.txt"
    if [[ ! -f "$TINY_GLOVE" ]]; then
      $PYTHON - <<'PY'
import pathlib
path = pathlib.Path('outputs/glove_tiny_100d.txt')
vec100_a = ' '.join(['0.1'] * 100)
vec100_b = ' '.join(['0.2'] * 100)
vec100_c = ' '.join(['0.3'] * 100)
lines = [
    f"good {vec100_a}\n",
    f"bad {vec100_b}\n",
    f"movie {vec100_c}\n",
    f"excellent {vec100_a}\n",
    f"awful {vec100_b}\n",
]
path.write_text(''.join(lines), encoding='utf-8')
print(f"Wrote {path} ({len(lines)} tokens, 100d)")
PY
    fi
    GLOVE_PATH="$TINY_GLOVE"
  fi
else
  if [[ -z "$GLOVE_PATH" ]]; then
    echo "ERROR: Set GLOVE_PATH to a GloVe file (e.g. data/glove.6B.100d.txt)" >&2
    exit 1
  fi
fi

$PYTHON -m src.rnn_imdb.train --config configs/imdb_rnn.json --glove_path "$GLOVE_PATH" $QUICK_FLAG
$PYTHON -m src.rnn_imdb.evaluate --config configs/imdb_rnn.json --glove_path "$GLOVE_PATH" $QUICK_FLAG
# Part C: FashionMNIST
$PYTHON -m src.fashion.train --config configs/fashion_cnn.json $QUICK_FLAG
$PYTHON -m src.fashion.eval_test --config configs/fashion_cnn.json $QUICK_FLAG
# Make report
$PYTHON -m src.make_report
