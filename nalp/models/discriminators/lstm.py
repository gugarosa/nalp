"""Long Short-Term Memory discriminator.
"""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, LSTMCell

from nalp.core import Discriminator
from nalp.utils import logging

logger = logging.get_logger(__name__)


class LSTMDiscriminator(Discriminator):
    """A LSTMDiscriminator class is the one in charge of a
    discriminative Long Short-Term Memory implementation.

    References:
        S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

    """

    def __init__(
        self, embedding_size: int = 32, hidden_size: int = 64
    ) -> None:
        """Initialization method.

        Args:
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        logger.info("Overriding class: Discriminator -> LSTMDiscriminator.")

        super(LSTMDiscriminator, self).__init__(name="D_lstm")

        self.embedding = Dense(embedding_size, name="embedding")

        self.cell = LSTMCell(hidden_size, name="lstm_cell")

        self.rnn = RNN(
            self.cell, name="rnn_layer", return_sequences=True, stateful=True
        )

        self.out = Dense(1, name="out")

        logger.info("Class overrided.")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        x = self.embedding(x)
        x = self.rnn(x)
        x = self.out(x)

        return x
