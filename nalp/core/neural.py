import tensorflow as tf
import tensorflow as tf

class Neural:

    def __init__(self, step_size=2, n_class=7):
        self.x = tf.placeholder(tf.float32, [None, step_size, n_class], name='inputs')
        self.y = tf.placeholder(tf.float32, [None, n_class], name='labels')

    def loss(self):
        raise NotImplementedError

    def optimizer(self):
        raise NotImplementedError

        