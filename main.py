#! /usr/bin/env python

from learn_chess import *
from dataSet import DataSet

def print_state(state):
    board = state[:128].reshape(8,8,2)
    player = 'white' if all(state[128:130] == [1, 0]) else 'black'
    
    f, t = np.argmax(state[130:].reshape((2, 64)), axis=1)
    fro = (f / 8, f % 8)
    to  = (t / 8, t % 8)
    print player
    chess.print_highlight_move(board, fro, to)

def train_with_transcript(regr):
    transcript = load_transcript('data/transcript.npz', 10000)
    for i in xrange(10):
        e = regr.train_with_data(transcript)
        print "Match " + str(i) + ":\n" + "Errors: " +  str(e)

def train_mixed(regr):
    transcript = load_transcript('data/transcript.npz', 10000)
    for i, (transcript_states, transcript_labels) in enumerate(transcript):
        states, rew, logs = play(regr)
        labels = np.array(rew).reshape(-1, 1)
        st = np.concatenate([states, transcript_states])
        lb = np.concatenate([labels, transcript_labels])
        e = regr.train_one_match(st, lb)
        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(labels.shape[0])

def load_transcript(path, batch_size):
    f = np.load('data/transcript.npz')
    states = regr.gen_states(f['boards'], f['players'], f['froms'], f['tos'])
    labels = f['rewards'].reshape(-1, 1)
    transcript = DataSet(states, labels, 258, 1, batch_size)
    return transcript

if __name__ == '__main__':
    regr = Regressor((520,260), (tf.nn.relu, tf.nn.relu, tf.identity))

    for i in xrange(10):
        for j in xrange(10):
            states, rew, logs = play(regr)
            labels = np.array(rew).reshape(-1, 4096, 1).astype(np.float32)
            state_batch = np.array(states).astype(np.float32)
            e = regr.train_one_match(state_batch, labels)
            print "Match " + str(500*i+j) + ":\n" + "Error: " +  str(e.mean()) + "\tinstances in last match: " + str(labels.shape[0])
        regr.save('models/2hl_cross_entropy.e' + str(i) + '.npz')
