#! /usr/bin/env python

from itertools import product

import chess


def get_input(board, player):
    fro_cands = [(fx, fy) for (fx, fy) in product(xrange(8), repeat=2) if (board[fx, fy, 1] == player) and len(get_dest_candidates(board, (fx, fy))) > 0]

    fro_select = fro_cands.pop(0)
    to_cands = get_dest_candidates(board, fro_select)
    to_select  =  to_cands.pop(0)

    while True:
        chess.print_highlight_move(board, fro_select, to_select, to_cands)
        inp = raw_input()
        print "[H"
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
            suc, check = chess.move(board, fro, to, pidx)

            if check:
                won = pidx
                break

    print 'White' if won == 1 else 'Black' + " has won!"

if __name__ == '__main__':
    main([get_input]*2)
    print "[?25h"
