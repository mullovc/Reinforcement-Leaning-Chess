import numpy as np
import tensorflow as tf
from dataSet import DataSet
from itertools import product
from mlp_layer import Layer

class Regressor:

    def __init__(self, hidden_layers, activation_funcs):
        board_shape = (128,)
        out_shape = (1,)

        one_act = np.identity(64)
        self.actions = np.array(list(product(one_act, repeat=2))).reshape(-1, 128)

        in_shape  = (board_shape[0] + 2 + self.actions.shape[1],)
        self.initialize_layers(in_shape + hidden_layers + out_shape, activation_funcs)

        # build generator graph, generating every possible action for a fed state and returning top-k
        self.player = tf.placeholder(tf.float32, [2])
        self.state  = tf.placeholder(tf.float32, board_shape)
        self.tiled_state  = tf.tile(tf.reshape(self.state,  (1, -1)), [len(self.actions), 1])
        self.tiled_player = tf.tile(tf.reshape(self.player, (1, -1)), [len(self.actions), 1])
        self.inputs = tf.concat([self.tiled_state, self.tiled_player, self.actions], axis=1)

        self.Qvals = self.get_feed_forward(self.inputs)
        self.top_k = tf.placeholder(tf.int32, shape=[])
        _, self.Qmax = tf.nn.top_k(tf.reshape(self.Qvals, [-1]), k=self.top_k)
        self.k_best = tf.gather(self.inputs, self.Qmax)


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

    #def train_with_transcript(self, boards, players, froms, tos, labels):
    def gen_states(self, boards, players, froms, tos):
        boards_flat = boards.reshape(-1, 128)
        fro = np.zeros((froms.shape[0], 64))
        to  = np.zeros((tos.shape[0],   64))
        fro[np.arange(len(froms)), [(8*x+y) for x, y in froms]] = 1
        to[np.arange(len(tos)),    [(8*x+y) for x, y in tos]]   = 1
        pl = np.array([[1, 0] if p == 1 else [0, 1] for p in players])

        return np.concatenate([boards_flat, pl, fro, to], axis=1)

    def k_best_actions(self, board_flat, k, player):
        p = [1, 0] if player == 1 else [0, 1]
        return self.sess.run(self.k_best, { self.state : board_flat, self.top_k : k, self.player : p })
        #return self.sess.run([self.Qmax, self.inputs], { self.state : board_flat, self.top_k : k, self.player : p })

    def index_to_action(self, idx):
        f, t = np.argmax(self.actions[idx].reshape((2, 64)), axis=1)
        fro = (f / 8, f % 8)
        to  = (t / 8, t % 8)
        return fro, to

    def input_to_action(self, inp):
        f, t = np.argmax(inp[130:].reshape((2, 64)), axis=1)
        fro = (f / 8, f % 8)
        to  = (t / 8, t % 8)
        return fro, to

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
        params = {}
        for i, l in enumerate(self.layers):
            W, b = self.sess.run([l.W, l.b])
            params['W' + str(i)] = W
            params['b' + str(i)] = b
        np.savez(fd, **params)

    def load(self, fd):
        params = np.load(fd)
        for i, l in enumerate(self.layers):
            self.sess.run([l.W.assign(params['W' + str(i)]), l.b.assign(params['b' + str(i)])])
