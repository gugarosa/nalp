"""Convolutional discriminator.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout

import nalp.utils.logging as l
from nalp.core import Discriminator

logger = l.get_logger(__name__)


class ConvDiscriminator(Discriminator):
    """A ConvDiscriminator class stands for the convolutional discriminative part
    of a Generative Adversarial Network.

    """

    def __init__(self, n_samplings=3, alpha=0.3, dropout_rate=0.3):
        """Initialization method.

        Args:
            n_samplings (int): Number of downsamplings to perform.
            alpha (float): LeakyReLU activation threshold.
            dropout_rate (float): Dropout activation rate.

        """

        logger.info('Overriding class: Discriminator -> ConvDiscriminator.')

        super(ConvDiscriminator, self).__init__(name='D_conv')

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a list for holding the convolutional layers
        self.conv = [Conv2D(
            64 * (i + 1), (5, 5), strides=(2, 2), padding='same', name=f'conv_{i}') for i in range(n_samplings)]

        # Defining a list for holding the dropout layers
        self.drop = [Dropout(dropout_rate, name=f'drop_{i}') for i in range(n_samplings)]

        # Defining the output as a logit unit that decides whether input is real or fake
        self.out = Dense(1, name='out')

        logger.info('Class overrided.')

    @property
    def alpha(self):
        """float: LeakyReLU activation threshold.

        """

        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    def call(self, x, training=True):
        """Method that holds vital information whenever this class is called.

        Args:
            x (tf.tensor): A tensorflow's tensor holding input data.
            training (bool): Whether architecture is under training or not.

        Returns:
            The same tensor after passing through each defined layer.

        """

        # For every possible convolutional and dropout layer
        for c, d in zip(self.conv, self.drop):
            # Applies the convolutional layer with a LeakyReLU activation and dropout
            x = d(tf.nn.leaky_relu(c(x), self.alpha), training=training)

        # Passing down the output layer
        x = self.out(x)

        return x
