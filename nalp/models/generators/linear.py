import tensorflow as tf
from tensorflow.keras import layers

import nalp.utils.logging as l
from nalp.core.model import Model

logger = l.get_logger(__name__)


class LinearGenerator(Model):
    """A LinearGenerator class stands for the linear generative part of a Generative Adversarial Network.

    """

    def __init__(self, input_shape, noise_dim, n_samplings, alpha):
        """Initialization method.

        Args:
            input_shape (tuple): An input shape for the tensor.
            noise_dim (int): Amount of noise dimensions.
            n_samplings (int): Number of upsamplings to perform.
            alpha (float): LeakyReLU activation threshold.

        """

        logger.info('Overriding class: Model -> LinearGenerator.')

        # Overrides its parent class with any custom arguments if needed
        super(LinearGenerator, self).__init__(name='G_linear')

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a property for the input noise dimension
        self.noise_dim = noise_dim

        # Defining a list for holding the linear layers
        self.linear = [layers.Dense(
            128 * (i + 1), name=f'linear_{i}') for i in range(n_samplings)]

        # Defining the output layer with a `tanh` activation for restraining interval to [-1, 1]
        self.out = layers.Dense(input_shape[0], activation='tanh', name='out')

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.Tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # For every possible linear layer
        for l in self.linear:
            # Applies the layer with a LeakyReLU activation
            x = tf.nn.leaky_relu(l(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x

