"""Bi-directional Long Short-Term Memory generator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, LSTMCell

from nalp.core import Generator
from nalp.encoders.integer import IntegerEncoder


class BiLSTMGenerator(Generator):
    """A BiLSTMGenerator class is the one in charge of a
    bi-directional Long Short-Term Memory implementation.

    References:
        S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

    """

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
    ) -> None:
        """Initialization method.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        super().__init__(name="G_bi_lstm")

        self.encoder = encoder

        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        cell_f = LSTMCell(hidden_size, name="lstm_cell_f")

        self.forward = RNN(
            cell_f, name="forward_rnn", return_sequences=True, stateful=True
        )

        cell_b = LSTMCell(hidden_size, name="lstm_cell_b")

        self.backward = RNN(
            cell_b,
            name="backward_rnn",
            return_sequences=True,
            stateful=True,
            go_backwards=True,
        )

        self.linear = Dense(vocab_size, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Method that holds vital information whenever this class is called.

        Args:
            x: A tensorflow's tensor holding input data.

        Returns:
            (tf.Tensor): The same tensor after passing through each defined layer.

        """

        x = self.embedding(x)

        x_f = self.forward(x)
        x_b = self.backward(x)

        x = tf.concat([x_f, x_b], -1)
        x = self.linear(x)

        return x
