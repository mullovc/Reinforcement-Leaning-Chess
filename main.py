#! /usr/bin/env python

import numpy as np

figures = {
        'pawn'    : 1,
        'rook'    : 2,
        'knight'  : 3,
        'bishop'  : 4,
        'queen'   : 5,
        'king'    : 6
        }
r_figures = {
        0 : 'none',
        1 : 'pawn',
        2 : 'rook',
        3 : 'knight',
        4 : 'bishop',
        5 : 'queen',
        6 : 'king'
        }

def validMove(fig, fro, to, color, occupied):
    if to[0] < 0 or to[1] < 0 or to[0] > 7 or to[1] > 7:
        return False, False

    if fig == 'pawn':
        forward = 1
        if color == 2:
            forward = -1
        if to[0] == fro[0] + forward and abs(fro[1] - to[1]) == 1 and occupied:
            return True, True
        if fro[1] != to[1]:
            return False, False
        if fro[0] == 1 and to[0] == 3:
            return True, False
        if fro[0] == 6 and to[0] == 4:
            return True, False
        if to[0] == fro[0] + forward:
            return True, False
    elif fig == 'rook':
        if not np.equal(fro, to).all() and (to[0] == fro[0] or to[1] == fro[1]):
            return True, True
    elif fig == 'knight':
        if (abs(fro[1] - to[1]) == 1 and abs(fro[0] - to[0]) == 3) or (abs(fro[0] - to[0]) == 1 and abs(fro[1] - to[1]) == 3):
            return True, True
    elif fig == 'bishop':
        if not np.equal(fro, to).all() and abs(fro[0] - to[0]) == abs(fro[1] - to[1]):
            return True, True
    elif fig == 'queen':
        if not np.equal(fro, to).all() and (to[0] == fro[0] or to[1] == fro[1]):
            return True, True
        if not np.equal(fro, to).all() and abs(fro[0] - to[0]) == abs(fro[1] - to[1]):
            return True, True
        #return validMove('bishop', fro, to, 0, occupied) or validMove('rook', fro, to, 0, occupied), True
    elif fig == 'king':
        if not np.equal(fro, to).all() and abs(fro[0] - to[0]) <= 1 and abs(fro[1] - to[1]) <= 1:
            return True, True
    return False, False


def build_board():
    board = np.zeros((8, 8, 2))
    for i in xrange(8):
        board[1, i, 0] = figures['pawn']
        board[0, i, 1] = 1
        board[1, i, 1] = 1
        board[6, i, 0] = figures['pawn']
        board[6, i, 1] = 2
        board[7, i, 1] = 2
    board[0, 0, 0] = figures['rook']
    board[0, 7, 0] = figures['rook']
    board[7, 0, 0] = figures['rook']
    board[7, 7, 0] = figures['rook']
    board[0, 1, 0] = figures['knight']
    board[0, 6, 0] = figures['knight']
    board[7, 1, 0] = figures['knight']
    board[7, 6, 0] = figures['knight']
    board[0, 2, 0] = figures['bishop']
    board[0, 5, 0] = figures['bishop']
    board[7, 2, 0] = figures['bishop']
    board[7, 5, 0] = figures['bishop']
    board[0, 3, 0] = figures['queen']
    board[7, 4, 0] = figures['queen']
    board[0, 4, 0] = figures['king']
    board[7, 3, 0] = figures['king']
    return board

def obstructed(board, fro, to):
    fig = board[fro[0], fro[1], 0]
    if fig == figures['knight']:
        return False
    dx = to[0] - fro[0]
    dy = to[1] - fro[1]
    sign = lambda x: 1 if x > 0 else -1
    diag = 1 if dx == dy else 0
    for i in xrange(sign(dx) * 1, dx, sign(dx)):
        if board[fro[0] + i, fro[1] + i * diag, 0] != 0:
            return True
    for i in xrange(sign(dy) * 1, dy, sign(dy)):
        if board[fro[0] + i * diag, fro[1] + i, 0] != 0:
            return True

def move(board, fro, to):
    fig = board[fro[0], fro[1]]
    fig_at = board[to[0], to[1]]
    occupied = fig_at[0] != 0
    if fig[1] == fig_at[1]:
        return False, False
    valid, beat = validMove(r_figures[fig[0]], fro, to, fig[1], occupied)

    if not valid or (occupied and not beat):
        return False, False
    if obstructed(board, fro, to):
        return False, False

    if fig_at[0] == figures['king']:
        return True, True

    board[to[0], to[1], 0] = board[fro[0], fro[1], 0]
    board[to[0], to[1], 1] = board[fro[0], fro[1], 1]
    board[fro[0], fro[1], 0] = 0
    board[fro[0], fro[1], 1] = 0
    return True, False


def print_board(board):
    fig_chars = {
            0 : ' ',
            1 : 'B',
            2 : 'T',
            3 : 'S',
            4 : 'L',
            5 : 'Q',
            6 : 'K',
            7 : '@'
            }

    colors = {
            (0, 0) : "[40m",
            (1, 0) : "[47m",
            (0, 1) : "[39;40m",
            (1, 1) : "[39;47m",
            (0, 2) : "[35;40m",
            (1, 2) : "[35;47m"
            }

    out = ''
    for lidx, l in enumerate(board.transpose((1, 0, 2))):
        for fidx, f in enumerate(l):
            out += colors[((lidx + fidx) % 2, int(f[1]))]
            out += fig_chars[f[0]]
        out += '[49m\n'
    print out

def main():
    board = build_board()
    #dbg_move_random()
    won = 0

    while not won:
        for player in xrange(1, 3):
            suc = False
            while not suc:
                fro, to = np.random.randint(8, size=[2,2])
                suc, check = move(board, fro, to)

            print_highlight_move(board, fro, to)
            
            if check:
                won = player
                break

            inp = raw_input()
            print '[11F'

    print "Player " + str(won) + " has won!"


def print_highlight_move(board, fro, to):
    fig_chars = {
            0 : ' ',
            1 : 'B',
            2 : 'T',
            3 : 'S',
            4 : 'L',
            5 : 'Q',
            6 : 'K',
            7 : '@'
            }

    colors = {
            (0, 0) : "[40m",
            (1, 0) : "[47m",
            (0, 1) : "[39;40m",
            (1, 1) : "[39;47m",
            (0, 2) : "[35;40m",
            (1, 2) : "[35;47m",
            (2, 0) : "[41m",
            (2, 0) : "[41m",
            (2, 1) : "[39;41m",
            (2, 1) : "[39;41m",
            (2, 2) : "[35;41m",
            (2, 2) : "[35;41m"
            }

    out = ''
    for lidx, l in enumerate(board.transpose((1, 0, 2))):
        for fidx, f in enumerate(l):
            f_col = 2 if (fro[0] == fidx and fro[1] == lidx) or (to[0] == fidx and to[1] == lidx) or f[0] == 7 else (lidx + fidx) % 2
            out += colors[(f_col, int(f[1]))]
            out += fig_chars[f[0]]
        out += '[49m\n'
    print out + "[39m"
    

def dbg_print_all_moves(board):
    for i in xrange(8):
        for j in xrange(8):
            b = np.copy(board)
            fro = [i, j]
            fig = b[i, j, 0]
            col = b[i, j, 1]
            if fig == 0:
                continue
            for k in xrange(8):
                for l in xrange(8):
                    to = [k, l]
                    if validMove(r_figures[fig], fro, to, col, False)[0] and not obstructed(board, fro, to):
                        b[k, l, 0] = 7
            #print_board(b)
            print_highlight_move(b, fro, fro)
            print '[11F'
            raw_input()
    for i in xrange(8):
        print ''

def dbg_do_all_moves():
    for i in xrange(8):
        for j in xrange(8):
            b = build_board()
            fro = [i, j]
            fig = b[i, j, 0]
            col = b[i, j, 1]
            if fig == 0:
                continue
            for k in xrange(8):
                for l in xrange(8):
                    to = [k, l]
                    if move(b, fro, to)[0]:
                        print_board(b)
                        b = build_board()
                        print '[11F'
                        raw_input()
    for i in xrange(8):
        print ''

def dbg_move_random():
    b = build_board()

    counter = 0
    inp = ''
    while inp != 'q':
        if inp == 's':
            inp = ''
            dbg_print_all_moves(b)
            counter = 0
        if inp == 'c':
            print b[:,:,1].transpose()
            inp = raw_input()
            counter = 0
        if inp == 'b':
            print b[:,:,0].transpose()
            inp = raw_input()
            counter = 0

        fro = np.random.randint(8, size=[2])
        to  = np.random.randint(8, size=[2])
        fig = b[fro[0], fro[0], 0]
        col = b[fro[0], fro[0], 1]
        counter += 1
        if counter >= 10000:
            inp = raw_input()
            counter = 0
        if fig == 0:
            continue
        if move(b, fro, to)[0]:
            #print_board(b)
            print_highlight_move(b, fro, to)
            print '[11F'
            counter = 0
            inp = raw_input()
    for i in xrange(8):
        print ''

if __name__ == '__main__':
    main()
