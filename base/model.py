##############################################################################
# This is file contains the Model class. The purpose of this class is to act #
# as an abstract class. All subclasses should inherit from Model.            #
##############################################################################

import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, sess, save_path, lr=0.001, restore_path=None):
        self.sess = sess
        self.save_path = save_path
        self.lr = lr

        self.build_network()

        self.saver = tf.train.Saver()

        if restore_path is not None:
            self.saver.restore(sess, restore_path)
        else:
            tf.global_variables_initializer().run()

    def build_network(self):
        raise NotImplementedError("build_network must be implemented in the subclass")

    def train(self, batch):
        raise NotImplementedError("train must be implemented in the subclass")

    def predict(self, frame):
        raise NotImplementedError("predict must be implemented in the subclass")

