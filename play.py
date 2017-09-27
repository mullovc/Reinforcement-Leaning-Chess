#! /usr/bin/env python

import sys
import curses
from itertools import product

import chess


def get_input(board, player):
    fro_cands = [(fx, fy) for (fx, fy) in product(xrange(8), repeat=2) if (board[fx, fy, 1] == player) and len(get_dest_candidates(board, (fx, fy))) > 0]

    fro_select = fro_cands.pop(0)
    to_cands = get_dest_candidates(board, fro_select)
    to_select  =  to_cands.pop(0)

    while True:
        print "[H"
        chess.print_highlight_move(board, fro_select, to_select, to_cands)
        inp = raw_input()
        if   inp == "j":
            fro_cands.append(fro_select)
            fro_select = fro_cands.pop(0)
            to_cands   = get_dest_candidates(board, fro_select)
            to_select  =  to_cands.pop(0)
        elif inp == "k":
            fro_cands.insert(0, fro_select)
            fro_select = fro_cands.pop()
            to_cands   = get_dest_candidates(board, fro_select)
            to_select  = to_cands.pop(0)
        elif inp == "h":
            to_cands.append(to_select)
            to_select = to_cands.pop(0)
        elif inp == "l":
            to_cands.insert(0, to_select)
            to_select = to_cands.pop()
        elif inp == "":
            return fro_select, to_select
        elif inp == "q":
            return None, None

def get_dest_candidates(board, fro):
    player = board[fro[0], fro[1], 1]
    return [(tx, ty) for (tx, ty) in product(xrange(8), repeat=2) if any(chess.move_possible(board, fro, (tx, ty), player))]


def main(players):
    assert len(players) == 2

    print "[H[J[?25l"
    board = chess.build_board()
    won = 0

    while not won:
        for pidx, player in zip([1, -1], players):
            fro, to = player(board, pidx)

            if to == None:
                won = 1 if pidx == -1 else -1
                break

            suc, check = chess.move(board, fro, to, pidx)

            if check:
                won = pidx
                break

    print ('White' if won == 1 else 'Black') + " has won!"

if __name__ == '__main__':
    opp_arg = sys.argv[1]
    player = get_input
    opponent = None
    if   opp_arg == 'human':
        opponent = get_input
    elif opp_arg == 'nn':
        from nn_opponent import NNOpponent
        opponent = NNOpponent(sys.argv[2])
    main([player, opponent])

    print "[?25h"
