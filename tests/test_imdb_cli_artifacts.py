
import os
import subprocess
import glob
import pytest
import sys

def test_cli_artifacts(tmp_path):
    # Use quick mode for speed
    vocab_path = tmp_path / "vocab.pkl"
    subprocess.run([
        sys.executable, "-m", "src.imdb.vocab", "--k", "1", "--n", "1", "--m", "5", "--quick", "--out", str(vocab_path)
    ], check=True)
    assert os.path.exists(vocab_path)
    # Train logreg
    subprocess.run([
        sys.executable, "-m", "src.imdb.train_classical", "--model", "logreg", "--quick", "--vocab", str(vocab_path)
    ], check=True)
    # Learning curves
    subprocess.run([
        sys.executable, "-m", "src.imdb.learning_curves", "--model", "logreg", "--quick", "--vocab", str(vocab_path)
    ], check=True)
    # Eval test
    subprocess.run([
        sys.executable, "-m", "src.imdb.eval_test", "--model", "logreg", "--quick", "--vocab", str(vocab_path)
    ], check=True)
    # Check artifacts
    assert len(glob.glob("outputs/tables/imdb_dev_gridsearch.csv")) == 1
    assert len(glob.glob("outputs/plots/imdb_learning_curves_logreg.png")) == 1
    assert len(glob.glob("outputs/tables/imdb_test_results_logreg.csv")) == 1
