"""Convolutional discriminator.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout

from nalp.core import Discriminator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class ConvDiscriminator(Discriminator):
    """A ConvDiscriminator class stands for the convolutional discriminative part
    of a Generative Adversarial Network.

    """

    def __init__(
        self,
        n_samplings: Optional[int] = 3,
        alpha: Optional[float] = 0.3,
        dropout_rate: Optional[float] = 0.3,
    ) -> None:
        """Initialization method.

        Args:
            n_samplings: Number of downsamplings to perform.
            alpha: LeakyReLU activation threshold.
            dropout_rate: Dropout activation rate.

        """

        logger.info("Overriding class: Discriminator -> ConvDiscriminator.")

        super(ConvDiscriminator, self).__init__(name="D_conv")

        self.alpha = alpha

        self.conv = [
            Conv2D(
                64 * (i + 1), (5, 5), strides=(2, 2), padding="same", name=f"conv_{i}"
            )
            for i in range(n_samplings)
        ]

        self.drop = [
            Dropout(dropout_rate, name=f"drop_{i}") for i in range(n_samplings)
        ]

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

        for c, d in zip(self.conv, self.drop):
            x = d(tf.nn.leaky_relu(c(x), self.alpha), training=training)

        x = self.out(x)

        return x
