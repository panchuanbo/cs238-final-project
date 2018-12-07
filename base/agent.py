################################################
# Base Agent for the Tetris AI.                #
# This part of the program interfaces directly #
# with the emulator. It should get the state   #
# of the board, current score (rewards), and   #
# control inputs to the emulator.              #
################################################

import os
import sys
import numpy as np

from nintaco import nintaco

from util.util import bcolors, kOrientations, MemAddr, Const

class Agent(object):
    """
    Base Agent class for various Tetris AI Agents
    This class acts as an abstract class. Please override this
    class with your own implementation.
    """

    def __init__(self, api, verbose=False):
        self.api = api
        self.verbose = verbose

        self.launched = False
        self.grid = np.zeros((Const.Board_Height, Const.Board_Width))

    def launch(self):
        """
        Starts the API and sets up the listeners.
        Throws an exception if this is called while the API is
        already running.
        """

        if not self.launched:
            self.api.addActivateListener(self._api_enabled)
            self.api.addFrameListener(self._frame_render_finished)
            self.api.addControllersListener(self._controller_listener)
            self.api.addAccessPointListener(self._piece_update, nintaco.PreWrite, 0x0042)
            self.api.run()
        else:
            raise Exception("Agent already running.")

    ### MARK: - Listeners ###

    def _frame_render_finished(self):
        raise NotImplementedError('Needs to be implemented in subclass')

    def _controller_listener(self):
        raise NotImplementedError('Needs to be implemented in subclass')

    def _piece_update(self, access_type, address, value):
        return value

    ### MARK: - Helper Methods ###

    def _print_board(self, board=None):
        """
        Prints the board (if verbose mode is on)
        """

        board = self.grid if board is None else board

        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                val = str(int(board[i,j]))
                if board[i,j] > 0:
                    print bcolors.FAIL + val + bcolors.ENDC,
                else:
                    print val,
            print ''

    def _count_total(self):
        """
        Counts the total number of pieces.
        """

        n_T = self.api.peekCPU16(MemAddr.T_Count)
        n_J = self.api.peekCPU16(MemAddr.J_Count)
        n_Z = self.api.peekCPU16(MemAddr.Z_Count)
        n_O = self.api.peekCPU16(MemAddr.O_Count)
        n_S = self.api.peekCPU16(MemAddr.S_Count)
        n_L = self.api.peekCPU16(MemAddr.L_Count)
        n_I = self.api.peekCPU16(MemAddr.I_Count)

        return n_T + n_J + n_Z + n_O + n_S + n_L + n_I

    def _get_score(self):
        """
        Get's the current score. The score is stored as HIG - MED - LOW
        where each represents a 2-digit base 10 numeral. Therefore, it needs
        to be reconstructed.
        """

        low = self.api.peekCPU(MemAddr.Score_Low)
        mid = self.api.peekCPU(MemAddr.Score_Mid)
        hig = self.api.peekCPU(MemAddr.Score_Hig)

        return hig * 10000 + mid * 100 + low

    def _count_holes(self, board):
        """
        Counts the number of 'holes' or 'gaps' in the board. A hole/gap is defined
        as a region of the board that cann't be accessed up/down/left/right from the top.
        This implementation is not perfect as there are some configurations that prevent
        pieces from fitting in, but it acts as a proxy for good board positioning.
        """

        visit = [(0, i) for i in range(Const.Board_Width) if board[0,i] == 0]
        viewed = set(visit)

        while len(visit) > 0:
            (i, j) = visit.pop()
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if abs(di) == abs(dj): continue
                    if i + di < 20 and i + di >= 0 and j + dj < 10 and j + dj >= 0:
                        if board[i+di, j+dj] == 0 and (i+di, j+dj) not in viewed:
                            visit.append((i + di, j + dj))
                            viewed.add((i + di, j + dj))

        return board.shape[0] * board.shape[1] - (np.count_nonzero(board) + len(viewed))

    def _determine_levelness(self, board):
        board[board == 2] = 1

        diff = 0
        prev = None
        for i in range(0, Const.Board_Width):
            cur = None
            for j in range(0, Const.Board_Height):
                if board[j][i] == 1:
                    cur = Const.Board_Height - j
                    break
            if cur is None:
                cur = 0
            if prev is not None:
                diff = np.abs(prev - cur) ** 2
            prev = cur

        return -diff

    def _simulate_piece_drop(self, piece_id):
        """
        Simulates taking a piece and dropping it completely given the
        current state of the board.
        """
        if piece_id == 19:
            return None

        (x, y) = (self.api.peekCPU(MemAddr.X_Loc), self.api.peekCPU(MemAddr.Y_Loc))
        piece = kOrientations[piece_id]
        r, c = np.argwhere(piece == 2)[0]

        board = np.array(self.grid)
        board[board == 2] = 0

        last_valid_location = y
        for i in range(y, 20):
            placed_correctly = True
            if i + piece.shape[0] > 20: break
            for cc in range(-c, piece.shape[1] - c):
                for rr in range(-r, piece.shape[0] - r):
                    if rr + i >= 0 and piece[rr + r, cc + c] > 0:
                        if board[rr + i, cc + x] == 1:
                            placed_correctly = False
            if placed_correctly:
                last_valid_location = i
            else:
                break

        for cc in range(-c, piece.shape[1] - c):
            for rr in range(-r, piece.shape[0] - r):
                if rr + last_valid_location >= 0 and piece[rr + r, cc + c] > 0:
                    board[rr + last_valid_location, cc + x] = 2

        return board

    def _update_board(self, piece_id):
        """
        Updates the board and places the current piece onto it.
        This method additionally computes some statistics about the current
        board, including the number of holes, number of empty spaces in occupied
        rows, and the the height of the highest column.
        """
        if piece_id == 19:
            return

        (x, y) = (self.api.peekCPU(MemAddr.X_Loc), self.api.peekCPU(MemAddr.Y_Loc))
        piece = kOrientations[piece_id]
        r, c = np.argwhere(piece == 2)[0]

        # Generates the board
        for addr in range(MemAddr.Board_Start, MemAddr.Board_End + 1):
            val = 0 if self.api.peekCPU(addr) >= 0xef else 1
            self.grid[(addr - MemAddr.Board_End) / 10, (addr - MemAddr.Board_Start) % 10] = val

        # Places the piece
        for cc in range(-c, piece.shape[1] - c):
            for rr in range(-r, piece.shape[0] - r):
                if rr + y >= 0 and piece[rr + r, cc + c] > 0:
                    self.grid[rr + y, cc + x] = 2

    def _count_empty(self, board):
        rows = np.sum(board, axis=1)
        n_empty = np.count_nonzero(rows) * 10 - np.sum(rows)

        return n_empty


    def _count_height(self, board, height_drop_piece=False):
        # Computes Height
        rows = np.sum(board, axis=1).reshape(20)
        if height_drop_piece:
            h_arr = [i for i in range(20) if rows[i] > 0]
            h_arr = [-1] if len(h_arr) == 0 else h_arr
            height = (Const.Board_Height - h_arr[0])
        else:
            h_arr = [i for i in range(20) if rows[i] == 0]
            h_arr = [-1] if len(h_arr) == 0 else h_arr
            height = (Const.Board_Height - h_arr[-1]) - 1

        return height

    def _print_transition(self, prev, action, cur, r):
        """
        Debug function that prints out the previous and current state
        concatenated with each other, along with useful information such as the
        action and reward information.
        """

        print 'Transitioning...'
        act = np.zeros((20, 3)) - 2
        act[0,1] = action
        act[1,1] = r
        transition = np.hstack((prev, act, cur))

        for i in range(transition.shape[0]):
            for j in range(transition.shape[1]):
                val = transition[i,j]
                if val == -2:
                    print '*',
                elif val != 0:
                    print bcolors.FAIL + str(int(val)) + bcolors.ENDC,
                else:
                    print int(val),
            print ''


    ### MARK: - Controller Metatdata ###

    def _api_enabled(self):
        print '------------------------'
        print '[%s] API Enabled' % (self.agent_name())
        self.display_agent_description()
        print '------------------------'

    def agent_name(self):
        raise NotImplementedError('Needs to be implemented in subclass')

    def display_agent_description(self):
        print 'No Description Given'
