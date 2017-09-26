#! /usr/bin/env python

import sys

from itertools import product
import numpy as np

import chess
from reward import calc_reward

def play():
    board = chess.build_board()
    won = 0

    actions = []
    reward_log = []
    round_counter = 0
    while not won:
        for p in [1, -1]:
            fro_cands = [(fx, fy) for (fx, fy) in product(xrange(8), repeat=2) if board[fx, fy, 1] == p]
            suc = False
            while not suc:
                fro = fro_cands[np.random.randint(len(fro_cands))]
                to = np.random.randint(8, size=(2))
                suc, check = chess.move(board, fro, to, p)
            actions.append((p, (np.copy(board), fro, to, p)))


            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 40:
                won = 3
                break


    rewards = None
    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
        rewards = calc_reward(actions, won, 0.96)
    elif won == 3:
        print "Took to many rounds"
    
    return won, actions, rewards, reward_log


def print_transcript(actions):
    for b, fro, to, p in actions:
        chess.print_highlight_move(b, fro, to)
        raw_input()
        print '[11F'

def gen_transcripts(n):
    boards  = []
    froms   = []
    tos     = []
    players = []
    winner  = []
    rewards = []
    valid_games  = 0
    valid_rounds = 0
    for x in xrange(n):
        print "Round " + str(x)
        w, a, r, l = play()
        if w in [1, -1]:
            valid_games  += 1
            valid_rounds += len(r)
            for p, (b, f, t, p) in a:
                boards.append(b)
                froms.append(f)
                tos.append(t)
                players.append(p)
            rewards.append(r)
    all_rewards = np.concatenate(rewards, axis=0)
    np.savez('data/transcript.npz', boards=boards, froms=froms, tos=tos, players=players, rewards=all_rewards)
    print "Total valid matches: " + str(valid_games)
    print "Total valid rounds:  " + str(valid_rounds)

if __name__ == '__main__':
    gen_transcripts(int(sys.argv[1]))
