#! /usr/bin/env python3

from pathlib import Path
import numpy as np
import h5py

import sys

sys.path.append(str(Path(__file__).parent.parent))

import board

FILE = Path("./data/dataset_80000_games.h5")


def convert(input_file, output_file):
    with h5py.File(input_file, "r") as f:
        data = f["data"][:10]
        labels = f["labels"][:10]

        o = data[:, :64].reshape(data.shape[0], 8, 8)

        # 64 is turn
        # 65 is castling
        # bit 3 is white kingside    0
        # bit 2 is black kingside    7
        # bit 1 is white queenside  56
        # bit 0 is black queenside  63

        # 4 bits for chess pieces - 1 byte

        o[:, 0, 0] = data[:, 65] * 14
        breakpoint()

        breakpoint()
        np.savez(output_file, data=data, labels=labels)


if __name__ == "__main__":
    convert(FILE, FILE.with_suffix(".npz"))
