import tensorflow as tf
from tensorflow.keras import layers

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


class GumbelSoftmax(layers.Layer):
    """A GumbelSoftmax class is the one in charge of a Gumbel-Softmax layer implementation.

    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax.
        Preprint arXiv:1611.01144 (2016).

    """

    def __init__(self, axis=-1, **kwargs):
        """Initialization method.

        Args:
            axis (int): Axis to perform the softmax operation.

        """

        # Overrides its parent class with any custom arguments if needed
        super(GumbelSoftmax, self).__init__(**kwargs)

        # Defining a property for holding the intended axis
        self.axis = axis

    def call(self, x, tau):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            tau (float): Gumbel-Softmax temperature parameter.

        Returns:
            Gumbel-Softmax output and its argmax token.

        """

        # Adds a sampled Gumbel distribution to the input
        x += gumbel_distribution(x.shape)

        # Applying the softmax over the Gumbel-based input
        x = tf.nn.softmax(x * tau, axis=self.axis)

        # Sampling an argmax token from the Gumbel-based input
        y = tf.stop_gradient(tf.argmax(x, axis=self.axis))

        return x, y

    def get_config(self):
        """Gets the configuration of the layer for further serialization.

        """

        # Defining a dictionary holding the configuration
        config = {'axis': self.axis}

        # Overring the base configuration
        base_config = super(GumbelSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
