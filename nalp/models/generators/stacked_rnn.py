# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Stacked Recurrent Neural Network generator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding, SimpleRNNCell

from nalp.core.model import Generator
from nalp.encoders.integer import IntegerEncoder


class StackedRNNGenerator(Generator):
    """Generate vocabulary logits with stacked stateful recurrent cells."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: tuple[int, ...] = (64, 64),
    ) -> None:
        """Initialize the token embedding, stacked recurrent cells, and vocabulary projection.

        Calls accept integer IDs shaped (batch_size, length) and return logits shaped (batch_size, length, vocab_size).
        Every cell retains its state across calls with the same batch size until reset_state is invoked.

        Reference: J. Elman. Finding structure in time. Cognitive science 14.2 (1990).

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: Amount of hidden neurons per cell.

        """

        super().__init__(name="G_stacked_rnn")

        self.encoder = encoder

        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        self.cells = [SimpleRNNCell(size, name=f"rnn_cell{i}") for (i, size) in enumerate(hidden_size)]

        self.rnn = RNN(self.cells, name="rnn_layer", return_sequences=True, stateful=True)

        self.linear = Dense(vocab_size, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.linear(x)

        return x
