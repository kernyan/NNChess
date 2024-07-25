#! /usr/bin/env python3

import torch
from board import Board
import train
import numpy as np
import copy

pos = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # starting
    "4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",  # white wins
    "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 w kq - 0 1",  # black wins
    # "5bk1/1p2q1pp/3n4/2p5/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R b KQ e3 0 1",
    # "5bk1/1p2q1pp/8/2p5/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R b KQ e3 0 1",
    # "5bk1/1p5p/8/2p3p1/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1",
    # "5bk1/1p5p/8/6p1/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1",
    # "6k1/1p5p/8/6p1/1P2P3/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1"
]

# CHOICE = "p_linear_1mil"
CHOICE = "p_conv2d_small"
MODEL = f"model_dataset_56000_{CHOICE}_epoch_4"

if __name__ == "__main__":
    # b = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    model = train.model_list[CHOICE]()
    model.eval()
    model.load_state_dict(torch.load(f"./data/{MODEL}.pth"))

    for p in pos:
        b = Board(p)
        b1 = b.serialize()
        o = torch.from_numpy(b1).float()

        with torch.no_grad():
            output = model(o)
            print(output)
