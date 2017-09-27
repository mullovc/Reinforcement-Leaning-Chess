from regressor import Regressor
import tensorflow as tf
import chess

class NNOpponent():
    def __init__(self, model_path):
        self.k = 64
        self.regr = Regressor((520, 260), (tf.nn.relu, tf.nn.relu, tf.identity))
        self.regr.load(model_path)

    def __call__(self, board, player):
        return self.get_input(board, player)

    def get_input(self, board, player):
        board_flat = board.reshape([1, -1])
        top_k, inp = self.regr.k_best_actions(board_flat, self.k, player)

        moved = False
        for idx in top_k:
            i = inp[idx]
            fro, to = self.regr.input_to_action(i)
            moved, check = chess.move(board, fro, to, player)
            if moved:
                return fro, to

        if not moved:
            print "Couldn't find suitable move in top " + str(self.k) + " candidates. Forfeiting."
            return None, None
