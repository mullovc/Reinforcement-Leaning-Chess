#! /usr/bin/env python

from learn_chess import *

def print_state(state):
    board = state[:128].reshape(8,8,2)
    player = 'white' if all(state[128:130] == [1, 0]) else 'black'
    
    f, t = np.argmax(state[130:].reshape((2, 64)), axis=1)
    fro = (f / 8, f % 8)
    to  = (t / 8, t % 8)
    print player
    chess.print_highlight_move(board, fro, to)

def train_with_transcript(regr):
    f = np.load('data/transcript.npz')
    transcript_states = regr.gen_states(f['boards'], f['players'], f['froms'], f['tos'])
    transcript_labels = f['rewards'].reshape(-1, 1)
    for i in xrange(10):
        e = regr.train_one_match(transcript_states, transcript_labels)
        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(transcript_labels.shape[0])
    #regr.save('models/transcript_model.npz')

def train_mixed(regr):
    f = np.load('data/transcript.npz')
    transcript_states = regr.gen_states(f['boards'], f['players'], f['froms'], f['tos'])
    transcript_labels = f['rewards'].reshape(-1, 1)
    for i in xrange(100):
        states, rew, logs = play(regr)
        labels = np.array(rew).reshape(-1, 1)
        st = np.concatenate([states, transcript_states])
        lb = np.concatenate([labels, transcript_labels])
        e = regr.train_one_match(st, lb)
        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(labels.shape[0])


if __name__ == '__main__':
    regr = Regressor((258,128), (tf.nn.tanh, tf.nn.tanh, tf.nn.sigmoid))

    for i in xrange(300):
        states, rew, logs = play(regr)
        labels = np.array(rew).reshape(-1, 1).astype(np.float32)
        state_batch = np.array(states).astype(np.float32)
        e = regr.train_one_match(state_batch, labels)
        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(labels.shape[0])
