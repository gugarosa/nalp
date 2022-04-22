"""Embedded-text discriminator.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Embedding, MaxPool1D

from nalp.core import Discriminator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class EmbeddedTextDiscriminator(Discriminator):
    """A EmbeddedTextDiscriminator class stands for the
    text-discriminative part of a Generative Adversarial Network.

    """

    def __init__(
        self,
        vocab_size: Optional[int] = 1,
        max_length: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        n_filters: Optional[Tuple[int, ...]] = (64),
        filters_size: Optional[Tuple[int, ...]] = (1),
        dropout_rate: Optional[float] = 0.25,
    ) -> None:
        """Initialization method.

        Args:
            vocab_size: The size of the vocabulary.
            max_length: Maximum length of the sequences.
            embedding_size: The size of the embedding layer.
            n_filters: Number of filters to be applied.
            filters_size: Size of filters to be applied.
            dropout_rate: Dropout activation rate.

        """

        logger.info("Overriding class: Discriminator -> EmbeddedTextDiscriminator.")

        super(EmbeddedTextDiscriminator, self).__init__(name="D_text")

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        # Defining a list for holding the convolutional layers
        self.conv = [
            Conv2D(
                n,
                (k, embedding_size),
                strides=(1, 1),
                padding="valid",
                name=f"conv_{k}",
            )
            for n, k in zip(n_filters, filters_size)
        ]

        # Defining a list for holding the pooling layers
        self.pool = [
            MaxPool1D(max_length - k + 1, 1, name=f"pool_{k}") for k in filters_size
        ]

        # Defining a linear layer for serving as the `highway`
        self.highway = Dense(sum(n_filters), name="highway")

        # Defining the dropout layer
        self.drop = Dropout(dropout_rate, name="drop")

        # And finally, defining the output layer
        self.out = Dense(2, name="out")

        logger.info("Class overrided.")

    def call(self, x: tf.Tensor, training: Optional[bool] = True) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.
            training: Whether architecture is under training or not.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        # Passing down the embedding layer
        x = self.embedding(x)

        # Expanding the last dimension
        x = tf.expand_dims(x, -1)

        # Passing down the convolutional layers following a ReLU activation
        # and removal of third dimension
        convs = [tf.squeeze(tf.nn.relu(conv(x)), 2) for conv in self.conv]

        # Passing down the pooling layers per convolutional layer
        pools = [pool(conv) for pool, conv in zip(self.pool, convs)]

        # Concatenating all the pooling outputs into a single tensor
        x = tf.concat(pools, 2)

        # Calculating the output of the linear layer
        hw = self.highway(x)

        # Calculating the `highway` layer
        x = tf.math.sigmoid(hw) * tf.nn.relu(hw) + (1 - tf.math.sigmoid(hw)) * x

        # Calculating the output with a dropout regularization
        x = self.out(self.drop(x, training=training))

        return x
