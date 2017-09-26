from itertools import product

import numpy as np

import chess

#def calc_reward(log, won, rho):
#    if   won ==  1:
#        p1init_rew = 1
#        p2init_rew = 0.6
#    elif won == -1:
#        p1init_rew = 0.6
#        p2init_rew = 1
#    elif won in [3, 4]:
#        p1init_rew = 0.5
#        p2init_rew = 0.5
#    p1log = [(i, x) for (i, (p, x)) in enumerate(log) if p ==  1]
#    p2log = [(i, x) for (i, (p, x)) in enumerate(log) if p == -1]
#
#    p1rews = norec_reward(p1init_rew, p1log, rho)
#    p2rews = norec_reward(p2init_rew, p2log, rho)
#    all_rews = []
#    for i in xrange(len(log)):
#        if   p1rews and p1rews[-1][0] == i:
#            all_rews.append(p1rews.pop()[1])
#        elif p2rews and p2rews[-1][0] == i:
#            all_rews.append(p2rews.pop()[1])
#        else:
#            raise Exception
#
#    assert len(log) == len(all_rews)
#    return all_rews
#
#
#def norec_reward(init_rval, log, rho):
#    rew = []
#    rval_mult = init_rval
#
#    for i, rnd in log[::-1]:
#        rnd_rews = []
#        for (suc, check, valid, own, occ, same, beat, obst) in rnd:
#            c_rval = 1
#
#            if check:
#                c_rval = 1
#            elif suc and occ:
#                c_rval = 0.8
#            elif suc:
#                c_rval = 0.7
#            else:
#                if not own:
#                    c_rval *= 0.0
#                if not valid:
#                    c_rval *= 0.1
#                if not beat:
#                    c_rval *= 0.9
#                if same:
#                    c_rval *= 0.1
#                if obst:
#                    c_rval *= 0.5
#            rnd_rews.append(c_rval * rval_mult)
#        rval_mult *= rho
#        normalized_rew = np.array(rnd_rews) / sum(rnd_rews)
#        rew.append((i, normalized_rew))
#    return rew

def calc_reward(log, won, rho):
    if   won ==  1:
        p1base_rew = 2.0
        p2base_rew = 0.5
    elif won == -1:
        p1base_rew = 0.5
        p2base_rew = 2.0
    elif won in [3, 4]:
        p1base_rew = 0.4
        p2base_rew = 0.4
    p1log = [(i, x) for (i, (p, x)) in enumerate(log) if p ==  1]
    p2log = [(i, x) for (i, (p, x)) in enumerate(log) if p == -1]

    p1rews = norec_reward(p1base_rew, p1log, rho)
    p2rews = norec_reward(p2base_rew, p2log, rho)
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


def norec_reward(base_rval, log, rho):
    rew = []
    rval_mult = base_rval

    for i, rnd in log[::-1]:
        board, fro, to, player = rnd
        state_rew = reward_state(board, fro, to, player, rval_mult)
        rval_mult = ((rval_mult - 1) * rho) + 1
        rew.append((i, state_rew))
    return rew



def reward_state(board, fro, to, player, move_rwrd):
    rewards = np.zeros((8, 8, 8, 8))
    for fx, fy, tx, ty in product(xrange(8), repeat=4):
        suc, check = chess.move_possible(board, (fx, fy), (tx, ty), player)
        if not suc:
            continue

        c_rval = 1
        fig_at = board[tx, ty, 0]
        c_rval *= 1 + 0.1 * float(fig_at)

        if fro is not None and fro[0] == fx and fro[1] == fy and to[0] == tx and to[1] == ty:
            c_rval *= move_rwrd

        if check:
            c_rval *= 5
        rewards[fx, fy, tx, ty] = c_rval
    rewards = rewards / np.sum(rewards)

    return rewards
