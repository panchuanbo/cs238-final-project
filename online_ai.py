import os
import sys
import numpy as np

from nintaco import nintaco

from base.agent import Agent
from util.util import bcolors, kOrientations, MemAddr

from concurrent.futures import ThreadPoolExecutor

class OnlineAgent(Agent):
    def __init__(self, api, verbose=False):
        super(OnlineAgent, self).__init__(api, verbose=verbose)

        # noop, turn, left, right
        self.possible_moves = [-1, 0, 6, 7]
        self.placed_move = False
        self.show_board = False
        self.last_move = -2
        self.restart_game = 1
        self.game_restarted = True
        self.prev_x, self.prev_y = (-1, -1)
        self.prev_piece_id = -1

        self.e = ThreadPoolExecutor(max_workers=4)

    def _controller_listener(self):
        piece_id = self.api.peekCPU(0x0042)

        if not self.placed_move:
            # os.system('clear')
            print '--------------'
            move = self.__make_move() # TODO

            if self.restart_game > 0:
                self.api.writeGamepad(0, 3, True)
                self.restart_game -= 1
                move = -2
            else:
                if move >= 0:
                    self.api.writeGamepad(0, move, True)
            self.placed_move = True
            self.show_board = True

            if self.last_move != -2 and piece_id != 19:
                S  = self.grid.copy()
                self._update_board(self.api.peekCPU(0x0042))
                board = self._simulate_piece_drop(self.api.peekCPU(0x0042))
                n_empty = self._count_empty(self.grid)
                n_holes = self._count_holes(self.grid)
                height = self._count_height(board)
                levelness = self._determine_levelness(board)
                A  = self.last_move
                if height <= 2:
                    R = 1000
                else:
                    R = -200 * height
                R += -20 * n_holes + 10 * levelness # 10 * self._get_score()
                SP = self.grid.copy()

                self._print_transition(S, A, board, R)

            self.last_move = move
        else:
            self.placed_move = False

    def _frame_render_finished(self):
        """
        Renders the board the the current piece
        TODO: do this lazily, so we aren't calling read too often O_o
        """

        # To make things easier, we're going to modify the next piece drop
        # Always drop a certain type of block (currently square).
        # self.api.writeCPU(0x00bf, 0x0a)

        piece_id = self.api.peekCPU(0x0042)
        game_state = self.api.peekCPU(0x0048)

        # Restart the game
        if piece_id == 19 and (game_state == 10 or game_state == 0):
            self.game_restarted = True
            self.restart_game = 1
            self.prev_x, self.prev_y = (-1, -1)
            self.prev_piece_id = -1
            return

        # Probably a line clear... Skip
        if piece_id == 19 and game_state != 1:
            return

    def _piece_update(self, access_type, address, value):
        """
        Can be used to control the piece being dropped
        """
        return value

        if self.api.readCPU(0x0048) == 1:
            return 0x0a
        return value

    def __make_move(self, max_depth=8):
        (x, y) = (self.api.peekCPU(MemAddr.X_Loc), self.api.peekCPU(MemAddr.Y_Loc))
        piece_id = self.api.peekCPU(0x0042)
        clean_board = np.array(self.grid)
        clean_board[clean_board == 2] = 0

        if self.prev_x == x and self.prev_y == y and self.prev_piece_id == piece_id:
            return -1

        if piece_id == 19:
            return -1

        def move_helper(piece_id, x, y, action, depth, visited):
            (np, nx, ny) = self._get_piece_data_for_action(piece_id, x, y, action)
            nboard = self._place_piece_on_board(clean_board, np, nx, ny)

            if (np, nx, ny, depth) in visited:
                return visited[(np, nx, ny, depth)]
            if nboard is not None:
                # print (action, (np, nx, ny), (piece_id, x, y))
                nboard = self._simulate_piece_drop(np, use_board=nboard, x=nx, y=ny)
                # print action, nboard
                non_fill = self._count_row_non_fill(nboard)
                filled_rows = self._count_filled_line(nboard)
                n_empty = self._count_empty(nboard)
                n_holes = self._count_holes(nboard)
                height = self._count_height(nboard)
                levelness_ct_dw = self._determine_levelness(nboard, count_down=True)
                levelness_ct_up = self._determine_levelness(nboard, count_down=False)
                levelness = levelness_ct_dw + levelness_ct_up
                overhang = abs(levelness_ct_dw - levelness_ct_up)
                A  = self.last_move
                R = 0
                R -= 200 * height
                R -= 10 * levelness
                R -= 10 * overhang
                R -= non_fill
                R += filled_rows
                R -= n_empty

                if depth == 0:
                    visited[(np, nx, ny, depth)] = R
                    return R
                else:
                    Rs = [
                        move_helper(np, nx, ny, -1, depth-1, visited),
                        move_helper(np, nx, ny, 00, depth-1, visited),
                        move_helper(np, nx, ny, 06, depth-1, visited),
                        move_helper(np, nx, ny, 07, depth-1, visited)
                    ]
                    # print Rs
                    visited[(np, nx, ny, depth)] = R + max(Rs)
                    return visited[(np, nx, ny, depth)]
            else:
                return float('-inf')

        seen = {}

        # Experiment With TPE
        """
        f1 = self.e.submit(move_helper, piece_id, x, y, -1, max_depth, seen)
        f2 = self.e.submit(move_helper, piece_id, x, y, 00, max_depth, seen)
        f3 = self.e.submit(move_helper, piece_id, x, y, 06, max_depth, seen)
        f4 = self.e.submit(move_helper, piece_id, x, y, 07, max_depth, seen)

        actions = [f1.result(), f2.result(), f3.result(), f4.result()]

        return self.possible_moves[np.argmax(actions)]
        """

        actions = [
            move_helper(piece_id, x, y, -1, max_depth, seen),
            move_helper(piece_id, x, y, 00, max_depth, seen),
            move_helper(piece_id, x, y, 06, max_depth, seen),
            move_helper(piece_id, x, y, 07, max_depth, seen)
        ]

        return self.possible_moves[np.argmax(actions)]

    def agent_name(self):
        return 'Online Agent'

def main(args):
    nintaco.initRemoteAPI("localhost", 9999)
    api = nintaco.getAPI()

    agent = OnlineAgent(api)
    agent.launch()

if __name__ == "__main__":
    main(sys.argv)
