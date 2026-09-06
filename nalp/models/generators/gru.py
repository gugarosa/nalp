# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gated Recurrent Unit generator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, GRUCell

from nalp.core.model import Generator
from nalp.encoders.integer import IntegerEncoder


class GRUGenerator(Generator):
    """Generate vocabulary logits with a stateful gated recurrent unit."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
    ) -> None:
        """Initialize the token embedding, GRU, and vocabulary projection.

        Calls accept integer IDs shaped (batch_size, length) and return logits shaped (batch_size, length, vocab_size).
        Recurrent state persists across calls with the same batch size until reset_state is invoked.

        Reference: K. Cho, et al.
        Learning phrase representations using RNN encoder-decoder for statistical machine translation.
        Preprint arXiv:1406.1078 (2014).

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.

        """

        super().__init__(name="G_gru")

        self.encoder = encoder

        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        self.cell = GRUCell(hidden_size, name="gru")

        self.rnn = RNN(self.cell, name="rnn_layer", return_sequences=True, stateful=True)

        self.linear = Dense(vocab_size, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.linear(x)

        return x
