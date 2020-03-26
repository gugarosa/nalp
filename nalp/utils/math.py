import tensorflow as tf

def gumbel_distribution(input_shape, eps=1e-20):
    """
    """

    #
    u = tf.random.uniform(input_shape, 0, 1)

    return -tf.math.log(-tf.math.log(u + eps) + eps)
