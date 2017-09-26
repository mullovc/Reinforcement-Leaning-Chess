#! /usr/bin/env python

from learn_chess import *
from reward import calc_reward

def print_state(state):
    board = state[:128].reshape(8,8,2)
    player = 'white' if all(state[128:130] == [1, 0]) else 'black'
    
    f, t = np.argmax(state[130:].reshape((2, 64)), axis=1)
    fro = (f / 8, f % 8)
    to  = (t / 8, t % 8)
    print player
    chess.print_highlight_move(board, fro, to)

def train_with_transcript(regr):
    transcript = load_transcript('data/transcript.npz', 100)
    for i in xrange(10):
        for s, p, l in transcript:
            e = regr.train_one_match_inputs(s, p, l)
            print "Match " + str(i) + ":\n" + "Errors: " +  str(e)

def train_mixed(regr):
    transcript = load_transcript('data/transcript.npz', 100)
    for i, (transcript_states, transcript_labels) in enumerate(transcript):
        states, rew, logs = play(regr)
        labels = np.array(rew).reshape(-1, 1)
        st = np.concatenate([states, transcript_states])
        lb = np.concatenate([labels, transcript_labels])
        e = regr.train_one_match(st, lb)
        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(labels.shape[0])

def load_transcript(path, batch_size):
    f = np.load('data/transcript.npz')
    #states = regr.gen_states(f['boards'], f['players'], f['froms'], f['tos'])
    states = f['boards']
    player = np.array([[1, 0] if p == 1 else [0, 1] for p in f['players']])
    labels = f['rewards'].reshape(-1, 4096, 1)
    #transcript = DataSet(states, labels, 258, 1, batch_size)
    trimmed_states = states[:len(states) - (len(states) % batch_size)]
    trimmed_player = player[:len(player) - (len(player) % batch_size)]
    trimmed_labels = labels[:len(labels) - (len(labels) % batch_size)]
    batched_states = trimmed_states.reshape([-1, batch_size, 128])
    batched_player = trimmed_player.reshape([-1, batch_size,   2])
    batched_labels = trimmed_labels.reshape([-1, batch_size, 4096,   1])
    return zip(batched_states, batched_player, batched_labels)

if __name__ == '__main__':
    regr = Regressor((258, 128), (tf.nn.relu, tf.nn.relu, tf.identity))
    train_with_transcript(regr)

    for i in xrange(10):
        for j in xrange(10):
            winner, states, logs = play(regr)
            rewards = calc_reward(logs, winner, 0.96)
            labels = regr.tensor_to_labels(rewards)
            #labels = np.array(rew).reshape(-1, 4096, 1).astype(np.float32)
            state_batch = np.array(states).astype(np.float32)
            e = regr.train_one_match(state_batch, labels)
            print "Match " + str(500*i+j) + ":\n" + "Error: " +  str(e.mean()) + "\tinstances in last match: " + str(labels.shape[0])
