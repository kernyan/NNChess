#! /usr/bin/env python3

import sys
from flask import Flask, render_template, request
from pathlib import Path
import base64

FILE = Path(__file__).parent.resolve()
sys.path.append(str(FILE.parent))

from board import Board
import chess
import chess.svg

app = Flask(__name__)

STARTING_POS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = Board(STARTING_POS)

def to_svg(board: Board):
    image = chess.svg.board(board.board).encode("utf-8")
    return base64.b64encode(image).decode('utf-8')

@app.route('/', methods=('GET', 'POST'))
def index():
    inputs = {}
    reset = False
    if request.method == 'POST':
        inputs["move"] = request.form["move"]
        reset = request.form.get("action") == "reset"
        print(request.form.get("action"))
    imgtemp = '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>'
    imageV = {'image1':''}

    if len(inputs):
        print(inputs)
        try:
            chess_move = chess.Move.from_uci(inputs["move"])
            board.board.push(chess_move)
        except:
            print(f"{inputs} is not valid uci format")
        if reset:
            print("reset")
            board.board.reset()
        svg = to_svg(board)
        imageV['image1'] = imgtemp % svg
    return render_template('index.html', imgf=imageV)

if __name__ == '__main__':
    app.run()
