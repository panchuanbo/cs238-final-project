##################################################################
# This file contains the Deep-Q Network. It inherits from Model. #
##################################################################

import numpy as np
import tensorflow as tf

from base.model import Model

class DQN(Model):
    def __init__(self, sess, save_path, use_target=True, gamma=0.9999, n_actions=4, lr=0.0001, restore_path=None, hist_size=5):
        self.gamma = gamma
        self.n_actions = n_actions
        self.iteration = 0
        self.use_target = use_target
        self.hist_size = hist_size

        super(DQN, self).__init__(sess, save_path, lr=lr, restore_path=restore_path)

    def __create_placeholders(self):
        self.S = tf.placeholder(tf.float32, shape=[None, 20, 10, self.hist_size+1])
        self.A = tf.placeholder(tf.int32, shape=[None])
        self.R = tf.placeholder(tf.float32, shape=[None])
        self.S_n = tf.placeholder(tf.float32, shape=[None, 20, 10, self.hist_size+1])

    def __create_weights(self):
        # Weights
        initializer = tf.contrib.layers.xavier_initializer()
        self.W1 = tf.get_variable("W1", [4, 4, self.hist_size+1, 10], initializer=initializer)
        self.W2 = tf.get_variable("W2", [8, 8, 10, 10], initializer=initializer)
        self.W3 = tf.get_variable("W3", [2000, 20], initializer=initializer)
        self.W4 = tf.get_variable("W4", [20, 4], initializer=initializer)

        # Target Weights
        self.T_W1 = tf.get_variable("T_W1", [4, 4, self.hist_size+1, 10], initializer=initializer)
        self.T_W2 = tf.get_variable("T_W2", [8, 8, 10, 10], initializer=initializer)
        self.T_W3 = tf.get_variable("T_W3", [2000, 20], initializer=initializer)
        self.T_W4 = tf.get_variable("T_W4", [20, 4], initializer=initializer)

    # Build Network
    def build_network(self):
        self.__create_placeholders()
        self.__create_weights()

        # Network
        def gen_network(dat, w1, w2, w3, w4, strides=[1, 1, 1, 1]):
            a1 = tf.nn.conv2d(dat, w1, strides, padding='SAME')
            z1 = tf.nn.elu(a1)

            a2 = tf.nn.conv2d(z1, w2, strides, padding='SAME')
            z2 = tf.nn.elu(a2)

            # Flatten & FCL
            c3 = tf.contrib.layers.flatten(z2)

            a4 = tf.matmul(c3, w3)
            z4 = tf.nn.elu(a4)

            a5 = tf.matmul(z4, w4)

            return a5

        # In a DQN, the NN is an estimator for the Q value!
        self.Q = gen_network(self.S, self.W1, self.W2, self.W3, self.W4)
        if self.use_target:
            self.Q_n = gen_network(self.S_n, self.T_W1, self.T_W2, self.T_W3, self.T_W4)
        else:
            self.Q_n = gen_network(self.S_n, self.W1, self.W2, self.W3, self.W4)

        # Pred
        self.pred = tf.argmax(self.Q, axis=1)

        # Target and Pred Q Values
        indices = tf.one_hot(self.A, self.n_actions)
        self.target = self.R + self.gamma * tf.reduce_max(self.Q_n, axis=1)
        q_pred = tf.reduce_max(tf.multiply(indices, self.Q), axis=1)

        # Optimizer (w/ Clipping)
        self.loss = tf.reduce_sum(tf.square(tf.stop_gradient(self.target) - q_pred))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gvs = optimizer.compute_gradients(self.loss)
        print(type(gvs))
        gvs = [(grad, var) for (grad, var) in gvs if grad is not None]
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)

    def __update_target(self):
        W1_Op = self.T_W1.assign(self.W1)
        W2_Op = self.T_W2.assign(self.W2)
        W3_Op = self.T_W3.assign(self.W3)
        W4_Op = self.T_W4.assign(self.W4)

        self.sess.run([W1_Op, W2_Op, W3_Op, W4_Op])

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

        if self.iteration % 500 == 0:
            self.saver.save(self.sess, self.save_path)

        if self.use_target and self.iteration % 1000 == 0:
            self.__update_target()

        self.iteration += 1

    def predict(self, frame):
        (pred, Q) = self.sess.run([self.pred, self.Q], feed_dict={self.S: [frame]})

        print (pred, Q)
        return pred
