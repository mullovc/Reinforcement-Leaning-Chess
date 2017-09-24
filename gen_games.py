#! /usr/bin/env python

from itertools import product
import numpy as np

import chess
from learn_chess import calc_reward

def play():
    board = chess.build_board()
    won = 0

    actions = []
    reward_log = []
    round_counter = 0
    while not won:
        for p in [-1, 1]:
            fro_cands = [(fx, fy) for (fx, fy) in product(xrange(8), repeat=2) if board[fx, fy, 1] == p]
            suc = False
            while not suc:
                fro = fro_cands[np.random.randint(len(fro_cands))]
                to = np.random.randint(8, size=(2))
                suc, check = chess.move(board, fro, to, p)
            actions.append((np.copy(board), fro, to, p))

            fig    = board[fro[0], fro[1]]
            fig_at = board[to[0],  to[1]]
            same = fig[1] == fig_at[1]
            occ = board[to[0], to[1], 1] != 0
            valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
            obst = chess.obstructed(board, fro, to)
            reward_log.append((p, suc, check, valid, occ, same, beat, obst))


            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 40:
                won = 3

    rewards = calc_reward(reward_log, won, 0.95)

    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
    elif won == 3:
        print "Took to many rounds"
    
    return won, actions, rewards, reward_log

def print_transcript(actions):
    #print '[2;H[J'
    for b, fro, to, p in actions:
        chess.print_highlight_move(b, fro, to)
        raw_input()
        print '[11F'
        #print '[2;H'


if __name__ == '__main__':
    boards  = []
    froms   = []
    tos     = []
    players = []
    rewards = []
    valid_games  = 0
    valid_rounds = 0
    for x in xrange(10000):
        print "Round " + str(x)
        w, a, r, l = play()
        if w in [-1, 1]:
            valid_games  += 1
            valid_rounds += len(r)
            boards  += [b for b, _, _, _ in a]
            froms   += [f for _, f, _, _ in a]
            tos     += [t for _, _, t, _ in a]
            players += [p for _, _, _, p in a]
            rewards += r
    np.savez('data/transcript.npz', boards=boards, froms=froms, tos=tos, players=players, rewards=rewards)
    print "Total valid matches: " + str(valid_games)
    print "Total valid rounds:  " + str(valid_rounds)
    #print_transcript(a)
