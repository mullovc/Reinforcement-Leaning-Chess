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
        self.actions = np.array(list(product(one_act, repeat=2)), dtype=np.float32).reshape(-1, 128)

        in_shape  = (board_shape[0] + 2 + self.actions.shape[1],)
        self.initialize_layers(in_shape + hidden_layers + out_shape, activation_funcs)

        # build generator graph, generating every possible action for a fed state and returning top-k
        tile_n = self.actions.shape[0]
        self.player = tf.placeholder(tf.float32, (None, 2))
        self.state  = tf.placeholder(tf.float32, (None,) + board_shape)
        # tile actions according to batch size
        #self.tiled_actions = tf.tile(np.expand_dims(self.actions, 0), [tf.shape(None), 1, 1])
        self.tiled_actions = tf.tile(np.expand_dims(self.actions, 0), [tf.shape(self.state)[0], 1, 1])
        #self.tiled_actions = np.expand_dims(self.actions, 0)
        self.tiled_state   = tf.tile(tf.expand_dims(self.state,  1), [1, tile_n, 1])
        self.tiled_player  = tf.tile(tf.expand_dims(self.player, 1), [1, tile_n, 1])
        self.inputs = tf.concat([self.tiled_state, self.tiled_player, self.tiled_actions], axis=2)
        #self.player = tf.placeholder(tf.float32, (2))
        #self.state  = tf.placeholder(tf.float32, board_shape)
        #self.tiled_state  = tf.tile(tf.reshape(self.state,  (1, -1)), [len(self.actions), 1])
        #self.tiled_player = tf.tile(tf.reshape(self.player, (1, -1)), [len(self.actions), 1])
        #self.inputs = tf.reshape(tf.concat([self.tiled_state, self.tiled_player, self.actions], axis=1), (1, len(self.actions), 258))

        self.Qvals = self.get_feed_forward(self.inputs)
        self.top_k = tf.placeholder(tf.int32, shape=[])
        _, self.Qmax = tf.nn.top_k(tf.transpose(self.Qvals, [0, 2, 1]), k=self.top_k)
        #self.gather_idx = tf.concat([tf.reshape(tf.range(64), shape=[-1, 1]), tf.reshape(self.Qmax, shape=[-1, 1])], axis=1)
        #self.k_best = tf.gather_nd(self.inputs, self.gather_idx)


        # build gradient descent graph, taking states and labels
        self.x = tf.placeholder(tf.float32, shape=(None, None) + in_shape)
        self.t = tf.placeholder(tf.float32, (None, None, self.topology[-1]))

        self.o = self.get_feed_forward(self.x)
        self.E = tf.nn.softmax_cross_entropy_with_logits(logits=self.o, labels=self.t, dim=1)
        self.train_step = tf.train.AdamOptimizer().minimize(self.E)


        self.E_inputs = tf.nn.softmax_cross_entropy_with_logits(logits=self.Qvals, labels=self.t, dim=1)
        self.train_step_inputs = tf.train.AdamOptimizer().minimize(self.E_inputs)

        # initialize session
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train_one_match(self, states, labels):
        e, _ = self.sess.run([self.E, self.train_step], {self.x : states, self.t : labels})
        return e

    def train_one_match_inputs(self, board, player, labels):
        e, _ = self.sess.run([self.E_inputs, self.train_step_inputs], { self.player : player, self.state : board, self.t : labels })
        return e

    def train_with_data(self, data_set):
        errs = []
        for states, labels in data_set:
            e, _ = self.sess.run([self.E, self.train_step], {self.x : states, self.t : labels})
            errs.append[e]
        return errs

    #def train_with_transcript(self, boards, players, froms, tos, labels):
    def gen_states(self, boards, players, froms, tos):
        boards_flat = boards.reshape(-1, 128)
        fro = np.zeros((froms.shape[0], 64))
        to  = np.zeros((tos.shape[0],   64))
        fro[np.arange(len(froms)), [(8*x+y) for x, y in froms]] = 1
        to[np.arange(len(tos)),    [(8*x+y) for x, y in tos]]   = 1
        pl = np.array([[1, 0] if p == 1 else [0, 1] for p in players])

        #return np.expand_dims(np.concatenate([boards_flat, pl, fro, to], axis=1), axis=0)
        return np.concatenate([boards_flat, pl, fro, to], axis=1)

    def k_best_actions(self, board_flat, k, player):
        p = [1, 0] if player == 1 else [0, 1]
        p = np.array(p).reshape([1, 2])
        #return self.sess.run(self.k_best, { self.state : board_flat, self.top_k : k, self.player : p })
        kb, inp = self.sess.run([self.Qmax, self.inputs], { self.state : board_flat, self.top_k : k, self.player : p })
        return kb.reshape([-1]), inp.reshape((-1, 258))
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

    def tensor_to_labels(self, rews):
        return np.array(rews).reshape([-1, 4096, 1])

    def initialize_layers(self, topology, activation_funcs):
        self.topology = topology
        in_n = self.topology[0]
        self.layers = []
        for out_n, f in zip(self.topology[1:], activation_funcs):
            W = tf.Variable(tf.random_normal((in_n, out_n), stddev=0.1, dtype=tf.float32, name='weight'))
            b = tf.Variable(tf.random_normal((1, out_n), dtype=tf.float32, name='bias'))
            self.layers.append((W, b, f))
            in_n = out_n

    def get_feed_forward(self, x):
        next_x = x
        for W, b, f in self.layers:
            # x.dim: batch_size x k x 258;   W.dim: 258x128;    y.dim: batch_size x k x 128
            y = tf.tensordot(next_x, W, axes=[[2], [0]]) + b
            #y = tf.matmul(x, self.W) + self.b
            next_x = f(y)
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
        for i, (Wg, bg, _) in enumerate(self.layers):
            W, b = self.sess.run([Wg, bg])
            params['W' + str(i)] = W
            params['b' + str(i)] = b
        np.savez(fd, **params)

    def load(self, fd):
        params = np.load(fd)
        for i, (W, b, _) in enumerate(self.layers):
            self.sess.run([W.assign(params['W' + str(i)]), b.assign(params['b' + str(i)])])
