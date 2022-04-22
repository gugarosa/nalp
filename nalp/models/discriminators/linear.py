"""Linear discriminator.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Dense

from nalp.core import Discriminator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class LinearDiscriminator(Discriminator):
    """A LinearDiscriminator class stands for the
    linear discriminative part of a Generative Adversarial Network.

    """

    def __init__(
        self, n_samplings: Optional[int] = 3, alpha: Optional[float] = 0.01
    ) -> None:
        """Initialization method.

        Args:
            n_samplings: Number of downsamplings to perform.
            alpha: LeakyReLU activation threshold.

        """

        logger.info("Overriding class: Discriminator -> LinearDiscriminator.")

        super(LinearDiscriminator, self).__init__(name="D_linear")

        # Defining a property for the LeakyReLU activation
        self.alpha = alpha

        # Defining a list for holding the linear layers
        self.linear = [
            Dense(128 * i, name=f"linear_{i}") for i in range(n_samplings, 0, -1)
        ]

        # Defining the output as a logit unit that decides whether input is real or fake
        self.out = Dense(1, name="out")

        logger.info("Class overrided.")

    @property
    def alpha(self) -> float:
        """LeakyReLU activation threshold."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def call(self, x: tf.Tensor, training: Optional[bool] = True) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        # For every possible linear layer
        for layer in self.linear:
            # Applies the layer with a LeakyReLU activation
            x = tf.nn.leaky_relu(layer(x), self.alpha)

        # Passing down the output layer
        x = self.out(x)

        return x
