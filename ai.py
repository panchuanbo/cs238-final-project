################################################
# Starting point for the Tetris AI.            #
# This part of the program interfaces directly #
# with the emulator. It should get the state   #
# of the board, current score (rewards), and   #
# control inputs to the emulator.              #
################################################

import sys
import numpy as np

from nintaco import nintaco

from util import bcolors, kOrientations

class Agent:
    def __init__(self, api, verbose=False):
        self.api = api
        self.launched = False
        self.grid = np.zeros((20, 10))
        self.placed_move = False
        self.ctr = 0
        self.restart_game = False
        self.verbose = verbose

    def launch(self):
        """
        Starts the API and sets up the listeners.
        Throws an exception if this is called while the API is
        already running.
        """

        if not self.launched:
            self.api.addActivateListener(self.__api_enabled)
            self.api.addFrameListener(self.__frame_render_finished)
            self.api.addControllersListener(self.__controller_listener)
            self.api.run()
        else:
            raise ValueError("Agent already running.")

    def __api_enabled(self):
        print '[Agent] API Enabled'

    def __controller_listener(self):
        if not self.placed_move:
            if self.restart_game > 0:
                self.api.writeGamepad(0, 3, True)
                self.restart_game -= 1
            else:
                self.api.writeGamepad(0, 0, True)

        self.placed_move = not self.placed_move
        self.ctr += 1

    def __frame_render_finished(self):
        """
        Renders the board the the current piece
        TODO: do this lazily, so we aren't calling read too often O_o
        """

        (x, y) = (self.api.peekCPU(0x0040), self.api.peekCPU(0x0041))
        piece_id = self.api.peekCPU(0x0042)

        piece = kOrientations[piece_id]
        r, c = np.argwhere(piece == 2)[0]

        # Restart the game
        if piece_id == 19:
            self.restart_game = 1
            return

        # Generates the board
        for addr in range(0x0400, 0x04c7 + 1):
            val = 0 if self.api.peekCPU(addr) >= 0xef else 1
            self.grid[(addr - 0x0400) / 10, (addr - 0x0400) % 10] = val

        # Places the piece
        for cc in range(-c, piece.shape[1] - c):
            for rr in range(-r, piece.shape[0] - r):
                if rr + y >= 0 and piece[rr + r, cc + c] > 0:
                    self.grid[rr + y, cc + x] = 2

        if self.verbose:
            self.__print_board()

    def __print_board(self):
        """
        Prints the board (if verbose mode is on)
        """

        print 'Render...'
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                val = str(int(self.grid[i,j]))
                if self.grid[i,j] > 0:
                    print bcolors.FAIL + val + bcolors.ENDC,
                else:
                    print val,
            print ''


def main(args):
    nintaco.initRemoteAPI("localhost", 9999)
    api = nintaco.getAPI()

    agent = Agent(api)
    agent.launch()

if __name__ == "__main__":
    main(sys.argv)
