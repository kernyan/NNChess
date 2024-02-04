#! /usr/bin/env python3

import chess
import struct

symbols = chess.PIECE_SYMBOLS[1:] # drop None
sym_dict = {x:i+1 for i, x in enumerate(([x.upper() for x in symbols] + symbols))}

def mb(v: int, size: int) -> bytes:
    l = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    return struct.pack(l[size], v)

class Board():
    def __init__(self, fen: str):
        self.fen = fen
        self.board = chess.Board(fen)

    def serialize(self) -> bytes:
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

        b = b''
        for p in range(64):
            piece = self.board.piece_at(p)
            symbol = sym_dict[piece.symbol()] if piece else 0
            b += mb(symbol, 1)

        b += mb(int(self.board.turn), 1)

        castling = []
        castling.append(str(int(self.board.has_kingside_castling_rights(1))))
        castling.append(str(int(self.board.has_kingside_castling_rights(0))))
        castling.append(str(int(self.board.has_queenside_castling_rights(1))))
        castling.append(str(int(self.board.has_queenside_castling_rights(0))))

        castling_value = int(''.join(castling), 2)
        b += mb(castling_value, 1)
        b += mb(self.board.halfmove_clock, 1)
        b += mb(self.board.fullmove_number, 1)
        return b

if __name__ == "__main__":
    b = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    o = b.serialize()
    breakpoint()


