import os
import sys
import subprocess
import pytest

def test_rnn_training_quick(tmp_path):
    # Fake glove
    glove_path = tmp_path / "glove.txt"
    with open(glove_path, "w") as f:
        f.write("good " + " ".join(["0.1"]*100) + "\n")
        f.write("bad " + " ".join(["0.2"]*100) + "\n")
        f.write("movie " + " ".join(["0.3"]*100) + "\n")
        f.write("excellent " + " ".join(["0.4"]*100) + "\n")
        f.write("awful " + " ".join(["0.5"]*100) + "\n")
    config_path = tmp_path / "imdb_rnn.json"
    with open(config_path, "w") as f:
        f.write('''{"embedding_dim": 100, "hidden_dim": 8, "num_layers": 1, "dropout": 0.1, "batch_size": 2, "max_len": 8, "lr": 0.01, "epochs": 2, "seed": 42}\n''')
    # Train quick
    subprocess.run([
        sys.executable, "-m", "src.rnn_imdb.train", "--config", str(config_path), "--glove_path", str(glove_path), "--quick"
    ], check=True)
    # Check artifacts
    assert os.path.exists("outputs/checkpoints/imdb_rnn_best.pt")
    assert os.path.exists("outputs/plots/imdb_rnn_loss_curves.png")
    # Eval quick
    subprocess.run([
        sys.executable, "-m", "src.rnn_imdb.evaluate", "--config", str(config_path), "--glove_path", str(glove_path), "--quick"
    ], check=True)
    assert os.path.exists("outputs/tables/imdb_rnn_test_results.csv")
