#! /usr/bin/env python

from itertools import product

import numpy as np
import tensorflow as tf

import main as chess

def loss(args):
    board_batch = args[:,   :128].reshape(-1, 8, 8, 2)
    fro_batch   = args[:,128:130].astype(np.int32)
    to_batch    = args[:,130:132].astype(np.int32)

    batch_loss = []
    for board, fro, to in zip(board_batch, fro_batch, to_batch):
        occ = board[to[0], to[1], 1] != 0
        moved, check = chess.move(board, fro, to)
        if check:
            batch_loss += [0]
            continue
        if moved:
            batch_loss += [30 - occ * 20]
            continue
            #return np.array([30 - occ * 20])
        fig    = board[fro[0], fro[1]]
        fig_at = board[to[0],  to[1]]
        same = fig[1] == fig_at[1]

        valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
        obst = chess.obstructed(board, fro, to)

        batch_loss += [120 - valid * 50 - beat * 50 + obst * 50 + same * 50]
        continue
        #return np.array([120 - valid * 50 - beat * 50 + obst * 50])
    return np.array(batch_loss).astype(np.float32)

def reward(board_flat, fro, to):
    board = board_flat.reshape(8, 8, 2)

    occ = board[to[0], to[1], 1] != 0
    moved, check = chess.move(board, fro, to)
    if check:
        return 0
    if moved:
        return 30 - occ * 20

    fig    = board[fro[0], fro[1]]
    fig_at = board[to[0],  to[1]]
    same = fig[1] == fig_at[1]

    valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
    obst = chess.obstructed(board, fro, to)

    return 120 - valid * 50 - beat * 50 + obst * 50 + same * 50

def play(a, Qmax, sess, actions):
    board = chess.build_board()
    board_flat = board.flatten()
    won = 0

    actions_rewards = []

    while not won:
        for player in xrange(1, 3):
            suc = False
            while not suc:
                aidx, estim = sess.run([a, Qmax], { state : board_flat })
                fro, to = actions[aidx].reshape((2, 2))
                suc, check = chess.move(board, fro, to)
                r = reward(board_flat, fro, to)
                actions_rewards.append((np.copy(board_flat), estim, r))

            if check:
                won = player
                break

    print "Player " + str(won) + " has won!"
    return actions_rewards

if __name__ == '__main__':
    board = chess.build_board()
    flat_board = board.flatten()
    batch_size = 1
    #out_shape = (4,)
    in_shape = (flat_board.shape[0] + 4,)
    out_shape = (1,)

    W = tf.random_normal(in_shape + out_shape)
    b = tf.random_normal((1,) + out_shape)

    #x = tf.placeholder(tf.float32, shape=(None,) + in_shape)
    #o = tf.matmul(x, W) + b

    state = tf.placeholder(tf.float32, flat_board.shape)
    actions = np.array(list(product(xrange(8), repeat=4)))
    inputs = tf.concat([tf.tile(tf.reshape(state, (1, -1)), [len(actions), 1]), actions], axis=1)
    Qvals = tf.matmul(inputs, W) + b
    a = tf.cast(tf.argmax(Qvals, axis=0)[0], tf.int32)
    Qmax = Qvals[a]
    
    #r = tf.py_func(loss, [x], tf.float32, stateful=False)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ac_re = play(a, Qmax, sess, actions)
