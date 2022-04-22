"""Gated Recurrent Unit generator.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, GRUCell

from nalp.core import Generator
from nalp.encoders.integer import IntegerEncoder
from nalp.utils import logging

logger = logging.get_logger(__name__)


class GRUGenerator(Generator):
    """A GRUGenerator class is the one in charge of a
    Gated Recurrent Unit implementation.

    References:
        K. Cho, et al.
        Learning phrase representations using RNN encoder-decoder for statistical machine translation.
        Preprint arXiv:1406.1078 (2014).

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        hidden_size: Optional[int] = 64,
    ) -> None:
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        logger.info("Overriding class: Generator -> GRUGenerator.")

        super(GRUGenerator, self).__init__(name="G_gru")

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        # Creates a GRU cell
        self.cell = GRUCell(hidden_size, name="gru")

        # Creates the RNN loop itself
        self.rnn = RNN(
            self.cell, name="rnn_layer", return_sequences=True, stateful=True
        )

        # Creates the linear (Dense) layer
        self.linear = Dense(vocab_size, name="out")

        logger.info("Class overrided.")

    @property
    def encoder(self) -> IntegerEncoder:
        """An encoder generic object."""

        return self._encoder

    @encoder.setter
    def encoder(self, encoder: IntegerEncoder) -> None:
        self._encoder = encoder

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        # Firstly, we apply the embedding layer
        x = self.embedding(x)

        # We need to apply the input into the first recurrent layer
        x = self.rnn(x)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
