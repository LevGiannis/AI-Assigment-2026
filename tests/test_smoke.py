
import os
import pandas as pd
import pytest
import subprocess
import sys

def test_smoke_script():
    # Run the smoke test as a module
    subprocess.run([
        sys.executable, "-m", "src.utils.smoke_test"
    ], check=True)
    assert os.path.exists("outputs/tables/smoke_metrics.csv")
    assert os.path.exists("outputs/plots/smoke_plot.png")
    # Check metrics file
    df = pd.read_csv("outputs/tables/smoke_metrics.csv")
    assert set(df.columns) == {"class", "precision", "recall", "f1", "support"}
    # Check plot file size
    assert os.path.getsize("outputs/plots/smoke_plot.png") > 0
