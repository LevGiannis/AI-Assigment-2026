import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.device import get_device
from src.utils.metrics import compute_metrics
from src.utils.io import save_table
from src.utils.plots import save_plot

def main():
    set_seed(123)
    logger = get_logger("smoke_test")
    device = get_device()
    logger.info(f"Using device: {device}")

    # Dummy deterministic data
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 2, 1, 1, 0, 2])
    labels = [0, 1, 2]
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    logger.info(f"Metrics: {metrics}")

    # Save metrics table
    df = pd.DataFrame({
        'class': labels,
        'precision': metrics['per_class']['precision'],
        'recall': metrics['per_class']['recall'],
        'f1': metrics['per_class']['f1'],
        'support': metrics['per_class']['support']
    })
    save_table(df, "outputs/tables/smoke_metrics.csv")
    logger.info("Saved metrics table to outputs/tables/smoke_metrics.csv")

    # Dummy plot
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0.5, 0.7, 0.9], marker='o')
    ax.set_title("Smoke Test Plot")
    save_plot(fig, "outputs/plots/smoke_plot.png")
    logger.info("Saved plot to outputs/plots/smoke_plot.png")

if __name__ == "__main__":
    main()
