
import os
import subprocess
import glob
import sys

def run_module(module, args=None):
    cmd = [sys.executable, "-m", module]
    if args:
        cmd += args
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"{module} failed: {result.stderr}"

def test_end_to_end_quick():
    # Part A: IMDB classic
    run_module("src.imdb.vocab", ["--k", "1", "--n", "1", "--m", "1", "--quick", "--out", "outputs/tables/imdb_vocab_ig.pkl"])
    # Bulletproof: check that vocab file is not empty (at least 1 feature)
    import pickle
    vocab_path = "outputs/tables/imdb_vocab_ig.pkl"
    assert os.path.exists(vocab_path), f"Vocab file not found: {vocab_path}"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    assert len(vocab) >= 1, "Vocab has zero features in quick mode!"
    run_module("src.imdb.train_classical", ["--model", "logreg", "--config", "configs/classical_grid.yaml", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    run_module("src.imdb.train_classical", ["--model", "bernoulli_nb", "--config", "configs/classical_grid.yaml", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    run_module("src.imdb.learning_curves", ["--model", "logreg", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    run_module("src.imdb.learning_curves", ["--model", "bernoulli_nb", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    run_module("src.imdb.eval_test", ["--model", "logreg", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    run_module("src.imdb.eval_test", ["--model", "bernoulli_nb", "--quick", "--vocab", "outputs/tables/imdb_vocab_ig.pkl"])
    # Part B: IMDB RNN
    # Create minimal fake GloVe file for quick mode
    glove_path = "outputs/fake_glove.txt"
    tokens = set(["good", "bad", "movie", "excellent", "awful", "nice", "boring"])
    with open(glove_path, "w") as f:
        for t in tokens:
            f.write(f"{t} " + " ".join(["0.1"]*100) + "\n")
    run_module("src.rnn_imdb.train", ["--config", "configs/imdb_rnn.json", "--quick", "--glove_path", glove_path])
    run_module("src.rnn_imdb.evaluate", ["--config", "configs/imdb_rnn.json", "--quick", "--glove_path", glove_path])
    # Part C: FashionMNIST
    run_module("src.fashion.train", ["--config", "configs/fashion_cnn.json", "--quick"])
    run_module("src.fashion.eval_test", ["--config", "configs/fashion_cnn.json", "--quick"])
    # Make report
    run_module("src.make_report")

    # Check all required artifacts
    required = [
        "outputs/plots/imdb_learning_curves_logreg.png",
        "outputs/plots/imdb_learning_curves_bernoulli_nb.png",
        "outputs/tables/imdb_test_results_logreg.csv",
        "outputs/tables/imdb_test_results_bernoulli_nb.csv",
        "outputs/plots/imdb_rnn_loss_curves.png",
        "outputs/tables/imdb_rnn_test_results.csv",
        "outputs/plots/fashion_loss_curves.png",
        "outputs/tables/fashion_test_results.csv",
        "report/assignment2_report.md"
    ]
    for path in required:
        matches = glob.glob(path)
        assert matches, f"Missing artifact: {path}"
