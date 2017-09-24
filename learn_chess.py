#! /usr/bin/env python

from itertools import product

import numpy as np
import tensorflow as tf

import chess
from dataSet import DataSet
from regressor import Regressor


def calc_reward(log, won, rho):
    if   won ==  1:
        p1init_rew = 1
        p2init_rew = 0.3
    elif won == -1:
        p1init_rew = 0.3
        p2init_rew = 1
    elif won == 3:
        p1init_rew = 0.3
        p2init_rew = 0.3
    p1log = [(i, x) for (i, x) in enumerate(log) if x[0] ==  1]
    p2log = [(i, x) for (i, x) in enumerate(log) if x[0] == -1]

    p1rews = norec_reward(p1init_rew, p1log, rho)
    p2rews = norec_reward(p2init_rew, p2log, rho)
    all_rews = []
    for i in xrange(len(log)):
        if   p1rews and p1rews[-1][0] == i:
            all_rews.append(p1rews.pop()[1])
        elif p2rews and p2rews[-1][0] == i:
            all_rews.append(p2rews.pop()[1])
        else:
            raise Exception

    assert len(log) == len(all_rews)
    return all_rews


def norec_reward(init_rval, log, rho):
    rew = []
    rval = init_rval

    for i, (p, suc, check, valid, own, occ, same, beat, obst) in log[::-1]:
        c_rval = rval
        if suc:
            rval = rval * rho

        if check:
            c_rval = 1
        elif suc and occ:
            c_rval = 0.6
        elif suc:
            c_rval = 0.4
        else:
            if not own:
                c_rval *= 0.0
            if not valid:
                c_rval *= 0.2
            if not beat:
                c_rval *= 0.8
            if same:
                c_rval *= 0.1
            if obst:
                c_rval *= 0.4
        rew.append((i, c_rval))
    return rew

def play(regr):
    board = chess.build_board()
    board_flat = board.reshape((-1))
    won = 0

    k = 16
    actions = []
    reward_log = []
    round_counter = 0
    while not won:
        for p in [-1, 1]:
            (estim, aidx), inp = regr.k_best_actions(board_flat, k, p)
            for i, a in zip(aidx.flatten(), estim.flatten()):
                fro, to = regr.index_to_action(i)
                suc, check = chess.move(board, fro, to, p)
                actions.append(np.copy(inp[i]))

                fig    = board[fro[0], fro[1]]
                fig_at = board[to[0],  to[1]]
                own = fig[1] == p
                same = fig[1] == fig_at[1]
                occ = board[to[0], to[1], 1] != 0
                valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
                obst = chess.obstructed(board, fro, to)
                reward_log.append((p, suc, check, valid, own, occ, same, beat, obst))

                if suc:
                    break
            if not suc:
                print "Couldn't find suitable move in top " + str(k) + " candidates"
                won = 3
            #chess.print_highlight_move(board, fro, to)

            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 100:
                won = 3

    rewards = calc_reward(reward_log, won, 0.9)

    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
    elif won == 3:
        print "Took to many rounds"
    
    return actions, rewards, reward_log

def print_state(state):
    board = state[:128].reshape(8,8,2)
    player = 'black' if all(state[128:130] == [1, 0]) else 'white'
    
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
    regr = Regressor((258,128), (tf.nn.tanh, tf.nn.sigmoid))

    train_mixed(regr)
    regr.save('models/2hl_mixed_tr.npz')
    
    #for _ in xrange(5):
    #    train_with_transcript(regr)
    #    for i in xrange(10):
    #        states, rew, logs = play(regr)
    #        labels = np.array(rew).reshape(-1, 1)
    #        e = regr.train_one_match(states, labels)
    #        print "Match " + str(i) + ":\n" + "Error: " +  str(e) + "\tinstances in last match: " + str(labels.shape[0])
