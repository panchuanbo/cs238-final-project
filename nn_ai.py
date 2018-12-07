import os
import sys
import numpy as np
import tensorflow as tf

from nintaco import nintaco

from base.agent import Agent
from dqn import DQN
from util.replay_buffer import ReplayBuffer
from util.util import bcolors, kOrientations

class NeuralNetworkAgent(Agent):
    def __init__(self, api, network_class, sess, save_path, history_size=15, restore_path=None, verbose=False, train=False, test=False):
        super(NeuralNetworkAgent, self).__init__(api, verbose=verbose)

        # currently 7500 w/ 1000

        # Network
        self.network = network_class(sess, save_path, restore_path=restore_path, hist_size=history_size)
        self.replay_buffer = ReplayBuffer(max_size=2500)
        self.train = train
        self.history_size = history_size

        # Internal
        self.launched = False
        self.placed_move = False
        self.ctr = 0
        self.restart_game = 1
        self.game_restarted = True
        self.show_board = False
        self.last_move = -2
        self.start_state = np.zeros((20, 10, 1))
        self.possible_moves = [-1, 0, 6, 7]
        self.training_begun = False if not test else True
        self.epsilon = 1. if not test else 0
        self.decay = 0.999
        self.test = test

        self.prev_states = [self.start_state] * self.history_size

    def _controller_listener(self):
        piece_id = self.api.peekCPU(0x0042)
        game_state = self.api.peekCPU(0x0048)

        if piece_id != 19 and game_state == 1:
            # Train
            if self.train and self.replay_buffer.size() > 250 and not self.test:
                batch = self.replay_buffer.sample(batch_sz=250)
                self.network.train(batch)
                self.training_begun = True

                self.epsilon *= self.decay
                if self.epsilon < 0.010:
                    self.epsilon = 0.010

        if not self.placed_move:# and (random_move >= 0 or self.restart_game > 0):
            # os.system('clear')
            print '--------------'
            is_random = False
            move = None
            if np.random.random() < self.epsilon or not self.training_begun:
                move = np.random.choice(self.possible_moves)
                is_random = True
            else:
                tensor = np.dstack([self.grid] + self.prev_states)
                pred = self.network.predict(tensor)[0]
                move = self.possible_moves[pred]

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
                print 'Random:', is_random
                S  = self.grid.copy()
                self._update_board(self.api.peekCPU(0x0042))
                board = self._simulate_piece_drop(self.api.peekCPU(0x0042))
                n_empty = self._count_empty(self.grid)
                n_holes = self._count_holes(self.grid)
                height = self._count_height(board)
                levelness = self._determine_levelness(board)
                A  = self.last_move
                # R  = self._count_total() + self._get_score() - n_empty
                #R = (-50 * height) + (-20 * n_holes) + (self._get_score())
                if height <= 2:
                    R = 1000
                else:
                    R = -200 * height
                R += -20 * n_holes + 10 * levelness # 10 * self._get_score()
                SP = self.grid.copy()

                self.prev_states.insert(0, S)

                print np.dstack(self.prev_states).shape

                self.replay_buffer.add(np.dstack(self.prev_states),
                                       self.possible_moves.index(A),
                                       R,
                                       np.dstack([SP] + self.prev_states[:self.history_size]))

                self.prev_states = self.prev_states[:self.history_size]

                print self.epsilon
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
        self.api.writeCPU(0x00bf, 0x0a)

        piece_id = self.api.peekCPU(0x0042)
        game_state = self.api.peekCPU(0x0048)

        # Restart the game
        if piece_id == 19 and (game_state == 10 or game_state == 0):
            self.prev_states = [self.start_state] * self.history_size
            self.game_restarted = True
            self.restart_game = 1
            return

        # Probably a line clear... Skip
        if piece_id == 19 and game_state != 1:
            return

    def _piece_update(self, access_type, address, value):
        """
        Can be used to control the piece being dropped
        """
        if self.api.readCPU(0x0048) == 1:
            return 0x0a
        return value

    def agent_name(self):
        return 'NeuralNetworkAgent'

def main(args):
    nintaco.initRemoteAPI("localhost", 9999)
    api = nintaco.getAPI()

    with tf.Session() as sess:
        # agent = NeuralNetworkAgent(api, DQN, sess, save_path='checkpoints/model.ckpt', restore_path='checkpoints/model.ckpt', verbose=False, test=True)
        agent = NeuralNetworkAgent(api, DQN, sess, save_path='checkpoints/model.ckpt', verbose=False, train=True)
        agent.launch()

if __name__ == "__main__":
    main(sys.argv)
