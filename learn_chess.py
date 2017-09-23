#! /usr/bin/env python

from itertools import product

import numpy as np
import tensorflow as tf

import main as chess

def loss(args):
    board_batch  = args[:,   :128].reshape(-1, 8, 8, 2)
    fro_batch    = args[:,128:130].astype(np.int32)
    to_batch     = args[:,130:132].astype(np.int32)
    player_batch = args[:,    133].astype(np.int32)

    batch_loss = []
    for board, fro, to, player in zip(board_batch, fro_batch, to_batch, player_batch):
        occ = board[to[0], to[1], 1] != 0
        moved, check = chess.move(board, fro, to, player)
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

def reward(board_flat, fro, to, player):
    board = board_flat.reshape(8, 8, 2)

    occ = board[to[0], to[1], 1] != 0
    moved, check = chess.move(board, fro, to, player)
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

def play(Qmax, inputs, actions, sess):
    board = chess.build_board()
    board_flat = board.flatten()
    won = 0

    actions_rewards = []

    while not won:
        for p in xrange(1, 3):
            (estim, aidx), inp = sess.run([Qmax, inputs], { state : board_flat, player : [p] })
            for i, a in zip(aidx.flatten(), estim.flatten()):
                fro, to = actions[i].reshape((2, 2))
                suc, check = chess.move(board, fro, to, p)
                #print suc, check
                r = reward(board_flat, fro, to, p)
                actions_rewards.append((np.copy(inp[i]), a, r))
                if suc:
                    break
            if not suc:
                print "Couldn't find suitable move in top 8192 candidates"
                won = 3

            chess.print_highlight_move(board, fro, to)

            if check:
                won = p
                break

    print "Player " + str(won) + " has won!"
    return actions_rewards

def save_model(name, W, b, sess):
    Wz, bz = sess.run([W, b])
    np.savez('models/' + name, W=Wz, b=bz)

if __name__ == '__main__':
    board = chess.build_board()
    flat_board = board.flatten()
    batch_size = 1
    #out_shape = (4,)
    in_shape = (flat_board.shape[0] + 5,)
    out_shape = (1,)

    W = tf.Variable(tf.random_normal(in_shape + out_shape))
    b = tf.Variable(tf.random_normal((1,) + out_shape))


    player = tf.placeholder(tf.float32, [1])
    state  = tf.placeholder(tf.float32, flat_board.shape)
    actions = np.array(list(product(xrange(8), repeat=4)))
    tiled_state  = tf.tile(tf.reshape(state,  (1, -1)), [len(actions), 1])
    tiled_player = tf.tile(tf.reshape(player, (1, -1)), [len(actions), 1])
    inputs = tf.concat([tiled_state, tiled_player, actions], axis=1)
    Qvals = tf.matmul(inputs, W) + b
    #a = tf.cast(tf.argmax(Qvals, axis=0)[0], tf.int32)
    #Qmax = Qvals[a]
    Qmax = tf.nn.top_k(tf.transpose(Qvals), k=8**4)
    

    #r = tf.py_func(loss, [x], tf.float32, stateful=False)
    x = tf.placeholder(tf.float32, shape=(None,) + in_shape)
    t = tf.placeholder(tf.float32, shape=(None,) + out_shape)
    o = tf.matmul(x, W) + b
    E = tf.reduce_sum(tf.square(t - o))

    train = tf.train.GradientDescentOptimizer(0.5).minimize(E)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(1):
        acre = play(Qmax, inputs, actions, sess)

        states = np.array([s for s, _, _ in acre])
        estims = np.array([e for _, e, _ in acre])
        labels = np.array([l for _, _, l in acre]).reshape(-1, 1)
        sess.run(train, { x : states, t : labels })

    save_model('model', W, b, sess)
