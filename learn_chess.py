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
    elif won in [3, 4]:
        p1init_rew = 0.5
        p2init_rew = 0.5
    p1log = [(i, x) for (i, (p, x)) in enumerate(log) if p ==  1]
    p2log = [(i, x) for (i, (p, x)) in enumerate(log) if p == -1]

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

    for i, rnd in log[::-1]:
        rnd_rews = []
        for (suc, check, valid, own, occ, same, beat, obst) in rnd:
            c_rval = 1

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
            rnd_rews.append(c_rval * rval_mult)
        rval_mult *= rho
        normalized_rew = np.array(rnd_rews) / sum(rnd_rews)
        rew.append((i, normalized_rew))
    return rew

def play(regr):
    board = chess.build_board()
    board_flat = board.reshape((-1))
    won = 0

    k = 8**2
    actions = []
    reward_log = []
    round_counter = 0
    while not won:
        for p in [1, -1]:
            top_k, inp = regr.k_best_actions(board_flat, k, p)
            actions.append(np.copy(inp))
            round_log = []
            for i in inp:
                fro, to = regr.input_to_action(i)
                suc, check = chess.move_possible(board, fro, to, p)

                fig    = board[fro[0], fro[1]]
                fig_at = board[to[0],  to[1]]
                own = fig[1] == p
                same = fig[1] == fig_at[1]
                occ = board[to[0], to[1], 1] != 0
                valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
                obst = chess.obstructed(board, fro, to)
                round_log.append((suc, check, valid, own, occ, same, beat, obst))
            reward_log.append((p, round_log))

            moved = False
            for i in top_k:
                fro, to = regr.input_to_action(i)
                moved, _ = chess.move(board, fro, to, p)
                if moved:
                    break

            if not moved:
                print "Couldn't find suitable move in top " + str(k) + " candidates"
                won = 3
                break

            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 100:
                won = 4

    rewards = calc_reward(reward_log, won, 0.98)

    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
    elif won == 4:
        print "Took to many rounds"
    
    return actions, rewards, reward_log
