import os
import glob
import pandas as pd
from datetime import datetime

def section(title):
    return f"\n# {title}\n\n"

def img(path, caption=None):
    cap = f"\n*{caption}*" if caption else ""
    return f"![{os.path.basename(path)}]({path}){cap}\n"

def table_from_csv(path):
    df = pd.read_csv(path)
    return df.to_markdown(index=False)

def main():
    out = []
    out.append(f"Assignment 2 Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    # Part A
    out.append(section("Μέρος Α: IMDB Classic"))
    out.append("## Vocabulary/IG")
    out.append(img("outputs/plots/imdb_vocab_ig.png", "Vocabulary IG"))
    out.append("## Learning Curves")
    for clf in ["logreg", "bernoulli_nb"]:
        out.append(img(f"outputs/plots/imdb_learning_curves_{clf}.png", f"Learning curves {clf}"))
    out.append("## Test Results")
    for clf in ["logreg", "bernoulli_nb"]:
        out.append(f"### {clf}")
        out.append(table_from_csv(f"outputs/tables/imdb_test_results_{clf}.csv"))
    # Part B
    out.append(section("Μέρος Β: IMDB RNN"))
    out.append("## Αρχιτεκτονική + Embeddings")
    out.append(img("outputs/plots/imdb_rnn_architecture.png", "RNN Architecture"))
    out.append("## Loss Curves")
    out.append(img("outputs/plots/imdb_rnn_loss_curves.png", "RNN Loss Curves"))
    out.append("## Test Results")
    out.append(table_from_csv("outputs/tables/imdb_rnn_test_results.csv"))
    # Part C
    out.append(section("Μέρος Γ: FashionMNIST CNN"))
    out.append("## CNN Αρχιτεκτονική")
    out.append(img("outputs/plots/fashion_cnn_architecture.png", "CNN Architecture"))
    out.append("## Loss Curves")
    out.append(img("outputs/plots/fashion_loss_curves.png", "FashionMNIST Loss Curves"))
    out.append("## Test Results")
    out.append(table_from_csv("outputs/tables/fashion_test_results.csv"))
    # Hyperparameters
    out.append(section("Hyperparameters & Selection"))
    out.append("See configs/*.json and classical_grid.yaml. Selected by dev set performance.")
    # Datasets
    out.append(section("Datasets & Splits"))
    out.append("IMDB: 20k train/dev/test, FashionMNIST: 60k/10k, see code for split details.")
    # Repro
    out.append(section("Αναπαραγωγή"))
    out.append("Run:\n\n    bash scripts/run_all.sh --quick\n    pytest -q\n")
    # Save
    os.makedirs("report", exist_ok=True)
    with open("report/assignment2_report.md", "w") as f:
        f.write("\n".join(out))

if __name__ == "__main__":
    main()
