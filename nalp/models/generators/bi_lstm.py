"""Bi-directional Long Short-Term Memory generator.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, LSTMCell

from nalp.core import Generator
from nalp.encoders.integer import IntegerEncoder
from nalp.utils import logging

logger = logging.get_logger(__name__)


class BiLSTMGenerator(Generator):
    """A BiLSTMGenerator class is the one in charge of a
    bi-directional Long Short-Term Memory implementation.

    References:
        S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

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

        logger.info("Overriding class: Generator -> BiLSTMGenerator.")

        super(BiLSTMGenerator, self).__init__(name="G_bi_lstm")

        # Creates a property for holding the used encoder
        self.encoder = encoder

        # Creates an embedding layer
        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        # Creates a forward LSTM cell
        cell_f = LSTMCell(hidden_size, name="lstm_cell_f")

        # And a orward RNN layer
        self.forward = RNN(
            cell_f, name="forward_rnn", return_sequences=True, stateful=True
        )

        # Creates a backward LSTM cell
        cell_b = LSTMCell(hidden_size, name="lstm_cell_b")

        # And a backward RNN layer
        self.backward = RNN(
            cell_b,
            name="backward_rnn",
            return_sequences=True,
            stateful=True,
            go_backwards=True,
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

        # Then, we pass it to the forward layer
        x_f = self.forward(x)

        # We also pass it to the backward layer
        x_b = self.backward(x)

        # Then, we concatenate both layers
        x = tf.concat([x_f, x_b], -1)

        # The input also suffers a linear combination to output correct shape
        x = self.linear(x)

        return x
