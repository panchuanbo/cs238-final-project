####################################################################
# The actual network(s) and Replay Buffer is located in this file. #
# It will be used during the training and prediction process.      #
####################################################################

import random

import tensorflow as tf

class ReplayBuffer(object):
    def __init__(self, max_size=1000):
        self.max_size = 1000
        self.buff = []

    def add(s, a, r, sp):
        if len(self.buff) >= self.max_size:
            self.buff.pop(0)

        self.buff.append((s, a, r, sp))

    def sample(batch_sz=100):
        return random.sample(self.buff, min(batch_sz, len(self.buff)))

    def size():
        return len(self.buff)

class DQN(object):
    def __init__(self, sess, gamma=0.99, n_actions=3, lr=0.001):
        self.sess = sess
        self.gamma = gamma
        self.n_actions = n_actions
        self.lr = lr
        build_network()

    # Build Network
    def build_network(self):
        # Placeholders
        self.S = tf.placeholder(tf.float32, shape=[None, 20, 10, 1])
        self.A = tf.placeholder(tf.float32, shape=[None, 1])
        self.R = tf.placeholder(tf.float32, shape=[None, 1])
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
            c3 = tf.reshape(z2, [None, -1])
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
        pred = tf.reduce_max(tf.multiply(indices, Q), axis=1)

        self.loss = tf.square(target - pred)

        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self, batch):
        S   = [b[0] for b in batch]
        A   = [b[1] for b in batch]
        R   = [b[2] for b in batch]
        S_n = [b[3] for b in batch]
        (loss, op) = sess.run([self.loss, self.train_op], feed_dict={
            self.S:   S,
            self.A:   A,
            self.R:   R,
            self.S_n: S_n
        })
        print 'Loss:', loss

    def pred(self, frame):
        pred = sess.run(self.pred, feed_dict={self.S: [frame]})
