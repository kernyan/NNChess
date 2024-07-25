#! /usr/bin/env python3

import chess.pgn
import numpy as np
from board import Board
import time
import traceback
import os
from pathlib import Path

DEBUG = os.getenv("DEBUG", False)


result = {
    "1/2-1/2": 0,  # draw
    "1-0": 1,  # white wins
    "0-1": -1,  # black wins
    "*": 2,  # ongoing
}

count = 0
game_count = 0
INIT_SIZE = 5_000_000
SAVE_NPZ = Path("./data/dataset_5mil_games.npz")

if __name__ == "__main__":
    data_dset = np.zeros(shape=(INIT_SIZE, 2, 8, 8), dtype="uint8")
    labels_dset = np.zeros(shape=(INIT_SIZE,), dtype="int8")
    prev_time = time.time()

    # https://lumbrasgigabase.com/en/download-in-pgn-format-en/
    with open("./data/lumbras_giga_base.pgn", "r") as f:
        try:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    game_count += 1
                    score = result[game.headers["Result"]]
                    if score == 2:
                        continue
                    if DEBUG:
                        print(game.board().fen(), score)
                    labels_dset[count] = score
                    data_dset[count] = Board(game.board().fen()).serialize()
                    count += 1
                    while game := game.next():
                        labels_dset[count] = score
                        data_dset[count] = Board(game.board().fen()).serialize()
                        count += 1
                        if DEBUG:
                            print(game.board().fen(), score)
                except UnicodeDecodeError:
                    pass

                if game_count % 1000 == 0:
                    current_time = time.time()
                    print(
                        f"Processed {game_count}:{count} in {current_time - prev_time} seconds."
                    )
                    prev_time = current_time
        except KeyboardInterrupt:
            print(f"Processed {count} games. Saving to file.")
            np.savez(SAVE_NPZ, data=data_dset[:count], labels=labels_dset[:count])
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print(f"Processed {count} games. Saving to file.")
            np.savez(SAVE_NPZ, data=data_dset[:count], labels=labels_dset[:count])
