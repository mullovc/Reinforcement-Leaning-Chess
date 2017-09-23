import tensorflow as tf
import numpy as np
from dataSet import DataSet

class Layer:

    def __init__(self, in_n, out_n, activation_function):
        self.W = tf.Variable(tf.random_normal((in_n, out_n), dtype=tf.float32, name='weight'))
        self.b = tf.Variable(tf.random_normal((1, out_n), dtype=tf.float32, name='bias'))
        self.f = activation_function


    def get_feed_forward(self, x):
        y = tf.matmul(x, self.W) + self.b
        return self.f(y)
