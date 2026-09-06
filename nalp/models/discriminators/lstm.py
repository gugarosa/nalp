# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Long Short-Term Memory discriminator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, LSTMCell

from nalp.core.model import Discriminator


class LSTMDiscriminator(Discriminator):
    """Discriminate vocabulary distributions with a stateful LSTM."""

    def __init__(self, embedding_size: int = 32, hidden_size: int = 64) -> None:
        """Initialize the dense embedding, LSTM, and per-token logit projection.

        Calls accept distributions shaped (batch_size, length, vocab_size) and return (batch_size, length, 1) logits.
        Hidden and cell states persist across calls with the same batch size.

        Reference: S. Hochreiter, Jürgen Schmidhuber. Long short-term memory. Neural computation 9.8 (1997).

        Args:
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        super().__init__(name="D_lstm")

        self.embedding = Dense(embedding_size, name="embedding")

        self.cell = LSTMCell(hidden_size, name="lstm_cell")

        self.rnn = RNN(self.cell, name="rnn_layer", return_sequences=True, stateful=True)

        self.out = Dense(1, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.out(x)

        return x
