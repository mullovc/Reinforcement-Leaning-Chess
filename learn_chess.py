from itertools import product

import numpy as np
import tensorflow as tf

import chess
from dataSet import DataSet
from regressor import Regressor


def play(regr):
    board = chess.build_board()
    board_flat = board.reshape((1, -1))
    won = 0

    k = 8**2
    states = []
    log    = []
    round_counter = 0
    while not won:
        for p in [1, -1]:
            top_k, inp = regr.k_best_actions(board_flat, k, p)
            states.append(np.copy(inp))
            round_log = []


            moved = False
            for idx in top_k:
                i = inp[idx]
                fro, to = regr.input_to_action(i)
                moved, check = chess.move(board, fro, to, p)
                if moved:
                    log.append((p, (np.copy(board), fro, to, p)))
                    break

            if not moved:
                log.append((p, (np.copy(board), None, None, p)))
                won = 3
                print "Couldn't find suitable move in top " + str(k) + " candidates"
                break

            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 100:
                won = 4

    if won == 1 or won == -1:
        print "Player " + str(won) + " has won!"
    elif won == 4:
        print "Took to many rounds"
    
    return won, states, log
