#! /usr/bin/env python3

import chess
import struct
import numpy as np
from numpy import typing

sym_dict = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}
# 0 is empty
# 13 is en passant valid
# 14 is white rook castling
# 15 is black rook castling

print(sym_dict)


def mb(v: int, size: int) -> bytes:
    l = {1: "B", 2: "H", 4: "I", 8: "Q"}
    return struct.pack(l[size], v)


class Board:
    def __init__(self, fen: str):
        self.fen = fen
        self.board = chess.Board(fen)

    def serialize(self) -> typing.NDArray:
        """
        8x8 position
        each position - KQRBNP = 6*2 = 4 bits

        board     64* 4 bits  = 256
        who moves     1 bit   =   8
        castling      4 bits  =   8
        half clock    6 bits (50) = 8
        full moves    8 bits (200) = 8

        261 + 14 = 275 bits min
        512 + 32 = 544 bits
        """

        state = np.zeros(64, dtype=np.uint8)

        for p in range(64):
            piece = self.board.piece_at(p)
            symbol = sym_dict[piece.symbol()] if piece else 0
            state[p] = symbol

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert state[0] == 4
            state[0] = 14
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert state[7] == 4
            state[7] = 14
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert state[56] == 10
            state[56] = 15
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert state[63] == 10
            state[63] = 15
        if self.board.ep_square:
            assert state[self.board.ep_square] == 0
            state[self.board.ep_square] = 13

        out = np.zeros(shape=(2, 8, 8), dtype=np.uint8)
        out[0] = state.reshape(8, 8)
        out[1] = self.board.turn == chess.WHITE

        # convert from 2,8,8 to 8,8,3
        channel_1 = out[0, :, :]
        channel_2 = out[1, :, :]
        channel_3 = channel_1  # repeat channel 1
        new_data = np.stack((channel_1, channel_2, channel_3), axis = -1)

        return new_data


if __name__ == "__main__":
    b = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    o = b.serialize()
