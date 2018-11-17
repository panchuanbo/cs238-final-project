###########################################################################
# This file contains the Replay Buffer class, which stores experience and #
# allows getting a random subset of the samples which can then be used    #
# for training.                                                           #
###########################################################################

import random

class ReplayBuffer(object):
    def __init__(self, max_size=1000):
        self.max_size = 1000
        self.buff = []

    def add(self, s, a, r, sp):
        if len(self.buff) >= self.max_size:
            self.buff.pop(0)

        self.buff.append((s, a, r, sp))

    def sample(self, batch_sz=100):
        return random.sample(self.buff, min(batch_sz, len(self.buff)))

    def get_last_state(self):
        return self.buff[-1][-1]

    def size(self):
        return len(self.buff)
