#! /usr/bin/env python3

import torch
from board import Board
import train
import numpy as np
import copy

pos = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # starting position
    "4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",  # white dominate wins
    "rnbqkbnr/pppppppp/8/8/8/8/8/4K3 w kq - 0 1",  # black dominate wins
    "rnb1k1nr/pppp1ppp/8/2b1p3/4P2q/2N5/PPPP1PPP/R1BQKBNR b KQkq - 0 1", # black 4 move check
    "r1bqkbnr/p1pp1ppp/1pn5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", # white 4 move check
    # "5bk1/1p2q1pp/3n4/2p5/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R b KQ e3 0 1",
    # "5bk1/1p2q1pp/8/2p5/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R b KQ e3 0 1",
    # "5bk1/1p5p/8/2p3p1/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1",
    # "5bk1/1p5p/8/6p1/1P2P1b1/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1",
    # "6k1/1p5p/8/6p1/1P2P3/2NB1N2/P1P2PPP/R1BQK2R w KQ g6 0 1"
]

#CHOICE = "convnet"
CHOICE = "resnet_883"
MODEL = f"model_dataset_56000_{CHOICE}"

if __name__ == "__main__":
    model = train.model_list[CHOICE]()
    model.eval()
    model.load_state_dict(torch.load(f"./data/{MODEL}.pth"))

    with torch.no_grad():
        o = torch.from_numpy(np.array([Board(p).serialize() for p in pos])).float()
        output = model(o)
        print(output)
