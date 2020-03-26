import tensorflow as tf

import nalp.utils.constants as c


def gumbel_distribution(input_shape):
    """Samples a tensor from a Gumbel distribution.

    Args:
        input_shape (tuple): Shape of tensor to be sampled.

    Returns:
        An input_shape tensor sampled from a Gumbel distribution.

    """

    # Samples an uniform distribution based on the input shape
    u = tf.random.uniform(input_shape, 0, 1)

    # Samples from the Gumbel distribution
    g = -tf.math.log(-tf.math.log(u + c.EPSILON) + c.EPSILON)

    return g
