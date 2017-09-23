import numpy as np
import tensorflow as tf
from dataSet import DataSet
from itertools import product
from mlp_layer import Layer

class Regressor:

    def __init__(self):
        board_shape = (128,)
        in_shape  = (board_shape[0] + 5,)
        out_shape = (1,)
        self.initialize_layers(in_shape + (128,) + out_shape, [tf.nn.relu, tf.identity])


        # build generator graph, generating every possible action for a fed state and returning top-k
        self.player = tf.placeholder(tf.float32, [1])
        self.state  = tf.placeholder(tf.float32, board_shape)
        self.actions = np.array(list(product(xrange(8), repeat=4)))
        self.tiled_state  = tf.tile(tf.reshape(self.state,  (1, -1)), [len(self.actions), 1])
        self.tiled_player = tf.tile(tf.reshape(self.player, (1, -1)), [len(self.actions), 1])
        self.inputs = tf.concat([self.tiled_state, self.tiled_player, self.actions], axis=1)

        self.Qvals = self.get_feed_forward(self.inputs)
        self.top_k = tf.Variable(8**4)
        self.Qmax = tf.nn.top_k(tf.transpose(self.Qvals), k=self.top_k)


        # build gradient descent graph, taking states and labels
        self.x = tf.placeholder(tf.float32, shape=(None,) + in_shape)
        self.t = tf.placeholder(tf.float32, (None, self.topology[-1]))

        self.o = self.get_feed_forward(self.x)
        self.E = tf.reduce_mean(tf.square(self.t - self.o))
        self.train_step = tf.train.AdamOptimizer().minimize(self.E)

        # initialize session
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train_one_match(self, states, labels):
        e, _ = self.sess.run([self.E, self.train_step], {self.x : states, self.t : labels})
        return e

    def k_best_actions(self, board_flat, player):
        return self.sess.run([self.Qmax, self.inputs], { self.state : board_flat, self.player : [player] })

    def index_to_action(self, idx):
        return self.actions[idx].reshape((2, 2))

    def initialize_layers(self, topology, activation_funcs):
        self.topology = topology
        in_n = self.topology[0]
        self.layers = []
        for out_n, f in zip(self.topology[1:], activation_funcs):
            l = Layer(in_n, out_n, f)
            self.layers.append(l)
            in_n = out_n

    def get_feed_forward(self, x):
        next_x = x
        for l in self.layers:
            next_x = l.get_feed_forward(next_x)
        return next_x


    def setTrainingSet(self, data):
        self.trainingSet = data

    def setValidationSet(self, data):
        self.validationSet = data

    def train(self, epochs, verbose=False):
        for i in range(epochs):
            print 'epoch ' + str(i + 1)
            self.trainOneEpoch()

            if verbose:
                print 'accuracy: ' + str(self.accuracy(self.validationSet))

    def trainOneEpoch(self):
        for (x,t) in self.trainingSet:
            self.sess.run(self.train_step, {self.x : x, self.t : t})

    def apply(self, batch):
        return self.sess.run(self.o, {self.x : batch})

    def evaluate(self, batches):
        return np.concatenate([self.apply(x) for x in batches])

    def accuracy(self, batches=None):
        if batches == None:
            batches = self.validationSet

        pred   = self.o
        labels = self.t

        acc = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labels - pred), axis=1)))

        summed_accuracy = 0
        n_samples = 0
        for x,t in batches:
            summed_accuracy += self.sess.run(acc, {self.x : x, self.t : t})
            n_samples += 1
        return summed_accuracy / n_samples

    def save(self, fd):
        weights = []
        biases  = []
        for l in self.layers:
            W, b = self.sess.run([l.W, l.b])
            weights.append(W)
            biases.append(b.reshape((-1)))
        np.savez(fd, W=weights, b=biases)

    def load(self, fd):
        f = np.load(fd)
        weights = f['W']
        biases  = f['b']
        for l, W, b in zip(self.layers, weights, biases):
            self.sess.run([l.W.assign(W), l.b.assign(b.reshape((1,-1)))])
