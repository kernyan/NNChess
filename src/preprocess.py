#! /usr/bin/env python3

import chess.pgn

result = {
    "1/2-1/2": 0, # draw
    "1-0": 1, # white wins
    "0-1": -1, # black wins
    "*": 2 # ongoing
}

if __name__ == "__main__":
    #max0 = -1
    with open("./data/DATABASE4U.pgn", "r") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                #turns = int(game.headers["PlyCount"]) // 2 + 1
                #if max0 < turns:
                #    max0 = turns
                #    print(max0)
                breakpoint()
                score = result[game.headers["Result"]]
                if score == 2: continue
                print(game.board().fen(), score)
                while game := game.next():
                    print(game.board().fen(), score)
            except UnicodeDecodeError:
                pass

