# Assignment 2: IMDB & FashionMNIST

## Quick Start

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh --quick
pytest -q
```

## Full Run

```sh
bash scripts/run_all.sh
```

## Verification

- To verify all outputs and report are generated:

```sh
bash scripts/run_all.sh --quick
pytest -q
```

### Expected Output Artifacts

- outputs/plots/imdb_learning_curves_logreg.png
- outputs/plots/imdb_learning_curves_bernoulli_nb.png
- outputs/tables/imdb_test_results_logreg.csv
- outputs/tables/imdb_test_results_bernoulli_nb.csv
- outputs/plots/imdb_rnn_loss_curves.png
- outputs/tables/imdb_rnn_test_results.csv
- outputs/plots/fashion_loss_curves.png
- outputs/tables/fashion_test_results.csv
- report/assignment2_report.md

## Structure

- src/: all code
- tests/: all tests (pytest)
- configs/: config files
- outputs/: generated artifacts
- report/: generated report
- scripts/: run scripts

## Notes

- All pipelines support --quick for fast smoke/e2e tests.
- All code runs as modules (python -m ...).
- No manual PYTHONPATH needed (pytest.ini included).
