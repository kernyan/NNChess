#! /usr/bin/env python3

import chess.pgn
import h5py
import numpy as np
from board import Board
import time
import traceback
import os

DEBUG = os.getenv("DEBUG", False)


result = {
    "1/2-1/2": 0, # draw
    "1-0": 1, # white wins
    "0-1": -1, # black wins
    "*": 2 # ongoing
}

count = 0
game_count = 0
INIT_SIZE = 10_000_000

def create_dataset():
    file = h5py.File("./data/dataset.h5", "w")
    data_dset = file.create_dataset("data", (INIT_SIZE, 68), maxshape=(None, 68), dtype="uint8")
    labels_dset = file.create_dataset("labels", (INIT_SIZE,), maxshape=(None,), dtype="int8")
    return data_dset, labels_dset, file

if __name__ == "__main__":
    data_dset, labels_dset, file = create_dataset()
    prev_time = time.time()

    with open("./data/DATABASE4U.pgn", "r") as f:
        try:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    game_count += 1
                    score = result[game.headers["Result"]]
                    if score == 2: continue
                    if DEBUG: print(game.board().fen(), score)
                    labels_dset[count] = score
                    data_dset[count] = np.frombuffer(Board(game.board().fen()).serialize(), dtype="uint8")
                    count += 1
                    while game := game.next():
                        labels_dset[count] = score
                        data_dset[count] = np.frombuffer(Board(game.board().fen()).serialize(), dtype="uint8")
                        count += 1
                        if DEBUG: print(game.board().fen(), score)
                except UnicodeDecodeError:
                    pass

                if game_count % 1000 == 0:
                    current_time = time.time()
                    print(f"Processed {game_count}:{count} in {current_time - prev_time} seconds.")
                    prev_time = current_time
        except KeyboardInterrupt:
            print(f"Processed {count} games. Saving to file.")
            file.close()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f"Processed {count} games. Saving to file.")
            file.close()
