#! /usr/bin/env python

import sys
import curses
from itertools import product

import chess

scr = None

def get_input(board, player):
    fro_cands = [(fx, fy) for (fx, fy) in product(xrange(8), repeat=2) if (board[fx, fy, 1] == player) and len(get_dest_candidates(board, (fx, fy))) > 0]

    fro_select = fro_cands.pop(0)
    to_cands = get_dest_candidates(board, fro_select)
    to_select  =  to_cands.pop(0)

    while True:
        print "[H"
        print chess.print_highlight_move(board, fro_select, to_select, to_cands)
        inp = scr.getch()
        if   inp == curses.KEY_DOWN:
            fro_cands.append(fro_select)
            fro_select = fro_cands.pop(0)
            to_cands   = get_dest_candidates(board, fro_select)
            to_select  =  to_cands.pop(0)
        elif inp == curses.KEY_UP:
            fro_cands.insert(0, fro_select)
            fro_select = fro_cands.pop()
            to_cands   = get_dest_candidates(board, fro_select)
            to_select  = to_cands.pop(0)
        elif inp == curses.KEY_RIGHT:
            to_cands.append(to_select)
            to_select = to_cands.pop(0)
        elif inp == curses.KEY_LEFT:
            to_cands.insert(0, to_select)
            to_select = to_cands.pop()
        elif inp == ord("\n"):
            return fro_select, to_select
        elif inp == ord("q"):
            return None, None

def get_dest_candidates(board, fro):
    player = board[fro[0], fro[1], 1]
    return [(tx, ty) for (tx, ty) in product(xrange(8), repeat=2) if any(chess.move_possible(board, fro, (tx, ty), player))]


def main(players):
    assert len(players) == 2

    print "[H[J"
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

def start_game(s):
    global scr
    scr = s
    curses.curs_set(0)

    scr.addstr( "Choose opponent: \n1) human\n2) neural network")
    c = scr.getch()

    player = get_input
    opponent = None
    if   c == ord('1'):
        opponent = get_input
    elif c == ord('2'):
        from nn_opponent import NNOpponent
        opponent = NNOpponent(sys.argv[2])
    else:
        print "Option not recognized"
        exit(1)

    main([player, opponent])

if __name__ == '__main__':
    curses.wrapper(start_game)
