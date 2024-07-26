#! /usr/bin/env python3

from pathlib import Path
import numpy as np

FILE = Path(__file__).parents[1] / "data/dataset_5mil_games.npz"
OUT_FILE = Path(__file__).parents[1] / "data/dataset_5mil_games_883.npz"


def process(input_file: Path, output_file: Path):
    input_2_8_8 = dict(np.load(input_file))
    data = input_2_8_8["data"]
    channel_1 = data[:, 0, :, :]
    channel_2 = data[:, 1, :, :]
    channel_3 = channel_1  # repeat channel 1
    new_data = np.stack((channel_1, channel_2, channel_3), axis = -1)
    print(f"converted from {data.shape} to {new_data.shape}")
    np.savez(output_file, data=new_data, labels=input_2_8_8["labels"])


if __name__ == "__main__":
    process(FILE, OUT_FILE)

