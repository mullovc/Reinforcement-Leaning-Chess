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

def calc_reward(log, won, rho):
    if won == 1:
        p1init_rew =  1
        p2init_rew = -1
    if won == 2:
        p1init_rew = -1
        p2init_rew =  1
    elif won == 3:
        p1init_rew = -1
        p2init_rew = -1
    p1log = [(i, x) for (i, x) in enumerate(log) if x[0] == 1]
    p2log = [(i, x) for (i, x) in enumerate(log) if x[0] == 2]

    p1rews = norec_reward(p1init_rew, p1log, rho)
    p2rews = norec_reward(p2init_rew, p2log, rho)
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


def norec_reward(init_rval, log, rho):
    rew = []
    rval = init_rval

    for i, (p, suc, check, valid, occ, same, beat, obst) in log[::-1]:
        c_rval = rval
        if suc:
            rval = rval * rho
        else:
            rval = rval

        if check:
            c_rval += 0.6
        elif suc and occ:
            c_rval += 0.4
        elif suc:
            c_rval += 0.15
        else:
            c_rval += valid*0.1 - same*0.3 + beat*0.15 - obst*0.2
        rew.append((i, c_rval))
    return rew

#def calc_reward(log, won, rho):
#    if won == 3:
#        return rec_reward(-1, log, rho)
#
#    p1log = [x for x in log if x[0] == 1]
#    p2log = [x for x in log if x[0] == 2]
#
#    if won == 1:
#        return rec_reward(1, p1log, rho) + rec_reward(-1, p2log, rho)
#    elif won == 2:
#        return rec_reward(-1, p1log, rho) + rec_reward(1, p2log, rho)
#
#
#def rec_reward(qval, log, rho):
#    if not log:
#        return []
#    p, suc, check, valid, occ, same, beat, obst = log.pop()
#
#    if suc:
#        n_qval = qval * rho
#    else:
#        n_qval = qval
#    c_qval = n_qval
#
#    if check:
#        c_qval += 0.6
#    elif suc and occ:
#        c_qval += 0.4
#    elif suc:
#        c_qval += 0.15
#    else:
#        c_qval += valid*0.1 - same*0.3 + beat*0.15 - obst*0.2
#    return rec_reward(n_qval, log, rho) + [c_qval]

def play(Qmax, inputs, actions, sess):
    board = chess.build_board()
    board_flat = board.flatten()
    won = 0

    actions_preds = []
    reward_log = []
    round_counter = 0
    while not won:
        for p in xrange(1, 3):
            (estim, aidx), inp = sess.run([Qmax, inputs], { state : board_flat, player : [p] })
            for i, a in zip(aidx.flatten(), estim.flatten()):
                fro, to = actions[i].reshape((2, 2))
                suc, check = chess.move(board, fro, to, p)
                #actions_preds.append((np.copy(inp[i]), a))
                actions_preds.append(np.copy(inp[i]))

                fig    = board[fro[0], fro[1]]
                fig_at = board[to[0],  to[1]]
                same = fig[1] == fig_at[1]
                occ = board[to[0], to[1], 1] != 0
                valid, beat = chess.validMove(chess.r_figures[fig[0]], fro, to, fig[1], occ)
                obst = chess.obstructed(board, fro, to)
                reward_log.append((p, suc, check, valid, occ, same, beat, obst))

                if suc:
                    break
            if not suc:
                print "Couldn't find suitable move in top 8192 candidates"
                won = 3
            #chess.print_highlight_move(board, fro, to)

            if check:
                won = p
                break

            round_counter += 1
            if round_counter >= 100:
                won = 3

    print len(reward_log)
    rewards = calc_reward(reward_log, won, 0.99)

    if won == 1 or won == 2:
        print "Player " + str(won) + " has won!"
    elif won == 3:
        print "Took to many rounds"
    
    return actions_preds, rewards

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

    train = tf.train.AdamOptimizer().minimize(E)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in xrange(200):
        #acpre, rew = play(Qmax, inputs, actions, sess)
        states, rew = play(Qmax, inputs, actions, sess)

        #states = np.array([s for s, _ in acpre])
        #estims = np.array([e for _, e in acpre])
        labels = np.array(rew).reshape(-1, 1)
        e, _ = sess.run([E, train], { x : states, t : labels })
        print e

    save_model('model', W, b, sess)
