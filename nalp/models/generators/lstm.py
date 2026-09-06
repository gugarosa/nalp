# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Long Short-Term Memory generator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, LSTMCell

from nalp.core.model import Generator
from nalp.encoders.integer import IntegerEncoder


class LSTMGenerator(Generator):
    """Generate vocabulary logits with a stateful long short-term memory network."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
    ) -> None:
        """Initialize the token embedding, LSTM, and vocabulary projection.

        Calls accept integer IDs shaped (batch_size, length) and return logits shaped (batch_size, length, vocab_size).
        Hidden and cell states persist across calls with the same batch size until reset_state is invoked.

        Reference: S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        super().__init__(name="G_lstm")

        self.encoder = encoder

        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        self.cell = LSTMCell(hidden_size, name="lstm_cell")

        self.rnn = RNN(self.cell, name="rnn_layer", return_sequences=True, stateful=True)

        self.linear = Dense(vocab_size, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.linear(x)

        return x
