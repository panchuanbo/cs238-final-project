##################################################################
# This file contains the Deep-Q Network. It inherits from Model. #
##################################################################

import numpy as np
import tensorflow as tf

from model import Model

class DQN(Model):
    def __init__(self, sess, save_path, gamma=0.995, n_actions=4, lr=0.001, restore_path=None):
        self.gamma = gamma
        self.n_actions = n_actions
        self.iteration = 0

        super(DQN, self).__init__(sess, save_path, lr=lr, restore_path=restore_path)

    # Build Network
    def build_network(self):
        # Placeholders
        self.S = tf.placeholder(tf.float32, shape=[None, 20, 10, 1])
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.S_n = tf.placeholder(tf.float32, shape=[None, 20, 10, 1])

        # Weights
        self.W1 = tf.get_variable("W1", [4, 4, 1, 5])
        self.W2 = tf.get_variable("W2", [4, 4, 5, 10])
        self.W3 = tf.get_variable("W3", [2000, 4])

        # Network
        def gen_network(dat, w1, w2, w3, strides=[1, 1, 1, 1], stop_gradient=False):
            if stop_gradient:
                w1 = tf.stop_gradient(w1)
                w2 = tf.stop_gradient(w2)
                w3 = tf.stop_gradient(w3)

            a1 = tf.nn.conv2d(dat, w1, strides, padding='SAME')
            z1 = tf.nn.relu(a1)

            a2 = tf.nn.conv2d(z1, w2, strides, padding='SAME')
            z2 = tf.nn.relu(a2)

            # Flatten & FCL
            c3 = tf.contrib.layers.flatten(z2)
            c4 = tf.matmul(c3, w3)

            return c4

        # In a DQN, the NN is an estimator for the Q value!
        self.Q = gen_network(self.S, self.W1, self.W2, self.W3)
        self.Q_n = gen_network(self.S_n, self.W1, self.W2, self.W3, stop_gradient=True)

        # Pred
        self.pred = tf.argmax(self.Q, axis=1)

        # Optimizer
        indices = tf.one_hot(self.A, self.n_actions)
        target = self.R + self.gamma * tf.reduce_max(self.Q_n, axis=1)
        pred = tf.reduce_max(tf.multiply(indices, self.Q), axis=1)

        self.loss = tf.reduce_sum(tf.square(target - pred))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, batch):
        S   = np.array([b[0] for b in batch])
        A   = np.array([b[1] for b in batch])
        R   = np.array([b[2] for b in batch])
        S_n = np.array([b[3] for b in batch])

        (loss, op) = self.sess.run([self.loss, self.train_op], feed_dict={
            self.S:   S,
            self.A:   A,
            self.R:   R,
            self.S_n: S_n
        })
        print 'Loss:', loss

        if self.iteration % 100 == 0:
            self.saver.save(self.sess, self.save_path)

    def predict(self, frame):
        pred = self.sess.run(self.pred, feed_dict={self.S: [frame]})

        return pred
