import os
import pandas as pd
from sklearn.model_selection import train_test_split

# For demo: use synthetic data for tests/quick runs

def load_imdb_data(path=None, quick=False):
    if quick or path is None:
        # Small synthetic dataset: 200 train, 100 test
        data = [
            ("good movie", 1), ("bad movie", 0), ("excellent", 1), ("awful", 0), ("nice", 1), ("boring", 0)
        ] * 50
        df = pd.DataFrame(data, columns=["text", "label"])
        train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df["label"])
        return train.reset_index(drop=True), test.reset_index(drop=True)
    else:
        # Real IMDB loading logic (placeholder)
        raise NotImplementedError("Real IMDB loading not implemented yet.")
