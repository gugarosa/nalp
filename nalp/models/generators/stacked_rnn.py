"""Stacked Recurrent Neural Network generator.
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, SimpleRNNCell

from nalp.core import Generator
from nalp.encoders.integer import IntegerEncoder
from nalp.utils import logging

logger = logging.get_logger(__name__)


class StackedRNNGenerator(Generator):
    """A StackedRNNGenerator class is the one in charge of
    stacked Recurrent Neural Networks implementation.

    References:
        J. Elman. Finding structure in time. Cognitive science 14.2 (1990).

    """

    def __init__(
        self,
        encoder: Optional[IntegerEncoder] = None,
        vocab_size: Optional[int] = 1,
        embedding_size: Optional[int] = 32,
        hidden_size: Optional[Tuple[int, ...]] = (64, 64),
    ) -> None:
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: Amount of hidden neurons per cell.

        """

        logger.info("Overriding class: Generator -> StackedRNNGenerator.")

        super(StackedRNNGenerator, self).__init__(name="G_stacked_rnn")

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        # Creating a stack of RNN cells
        self.cells = [
            SimpleRNNCell(size, name=f"rnn_cell{i}")
            for (i, size) in enumerate(hidden_size)
        ]

        # Creates the RNN loop itself
        self.rnn = RNN(
            self.cells, name="rnn_layer", return_sequences=True, stateful=True
        )

        # Creates the linear (Dense) layer
        self.linear = Dense(vocab_size, name="out")

        logger.debug("Number of cells: %d.", len(hidden_size))
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
