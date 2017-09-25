from itertools import product

import numpy as np
import tensorflow as tf

import chess
from dataSet import DataSet
from regressor import Regressor


def calc_reward(log, won, rho):
    if   won ==  1:
        p1init_rew = 1
        p2init_rew = 0.6
    elif won == -1:
        p1init_rew = 0.6
        p2init_rew = 1
    elif won == 3:
        p1init_rew = 0.6
        p2init_rew = 0.6
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
    rval_mult = init_rval

    for i, (p, suc, check, valid, own, occ, same, beat, obst) in log[::-1]:
        c_rval = 1
        if suc:
            rval_mult *= rho

        if check:
            c_rval = 1
        elif suc and occ:
            c_rval = 0.8
        elif suc:
            c_rval = 0.7
        else:
            if not own:
                c_rval *= 0.0
            if not valid:
                c_rval *= 0.1
            if not beat:
                c_rval *= 0.9
            if same:
                c_rval *= 0.1
            if obst:
                c_rval *= 0.5
        rew.append((i, c_rval * rval_mult))
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
        for p in [1, -1]:
            #aidx, inp = regr.k_best_actions(board_flat, k, p)
            inp = regr.k_best_actions(board_flat, k, p)
            print inp.shape
            for i in inp:
                fro, to = regr.input_to_action(i)
                suc, check = chess.move(board, fro, to, p)
                actions.append(np.copy(i))

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

    rewards = calc_reward(reward_log, won, 0.98)

    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
    elif won == 3:
        print "Took to many rounds"
    
    return actions, rewards, reward_log
