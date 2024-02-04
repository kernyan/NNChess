#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("./data/losses_dataset_80000_games.csv")

if __name__ == "__main__":
    data = pd.read_csv(CSV)
    plt.figure(figsize=(10, 5))
    plt.plot(data["train_losses"], label="Train Loss")
    plt.plot(data["val_losses"], label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()

    # save plt to file
    plt.savefig(CSV.with_suffix(".png"))
