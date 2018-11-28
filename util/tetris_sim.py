from util.util import kOrientations, Const

class TetrisEnvironment(object):
    def __init__(self, board, piece_id, x, y, drop_counter=3):
        self.board = board
        self.piece_id = piece_id
        self.x = x
        self.y = y

        self.drop_counter_default = drop_counter
        self.drop_counter = drop_counter

    def simulate_move(self, move):
        """
        move = -1: noop
        move = 01: rotate
        move = 06: left
        move = 07: right
        """

        self.drop_counter -= 1

        if self.drop_counter == 0:
            self.drop_counter = self.drop_counter_default
            self.y -= 1
