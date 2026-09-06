# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Relational Memory Core generator."""

import tensorflow as tf
from tensorflow.keras.layers import RNN, Dense, Embedding

from nalp.core.model import Generator
from nalp.encoders.integer import IntegerEncoder
from nalp.models.layers.relational_memory_cell import RelationalMemoryCell


class RMCGenerator(Generator):
    """Generate vocabulary logits with stateful relational memory."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        n_slots: int = 3,
        n_heads: int = 5,
        head_size: int = 10,
        n_blocks: int = 1,
        n_layers: int = 3,
    ) -> None:
        """Initialize the token embedding, relational memory, and vocabulary projection.

        Calls accept integer IDs shaped (batch_size, length) and return logits shaped (batch_size, length, vocab_size).
        Record a statically known batch size on each call and retain recurrent state until reset_state is invoked.
        Reset restores the cell's padded or truncated identity memory rather than zeroing it.

        Reference: A. Santoro, et al. Relational recurrent neural networks.
        Advances in neural information processing systems (2018).

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of attention and feed-forward refinement blocks.
            n_layers: Number of layers per feed-forward network.

        """

        super().__init__(name="G_rmc")

        self.encoder = encoder

        self.embedding = Embedding(vocab_size, embedding_size, name="embedding")

        self.cell = RelationalMemoryCell(n_slots, n_heads, head_size, n_blocks, n_layers, name="rmc_cell")

        self.rnn = RNN(self.cell, name="rnn_layer", return_sequences=True, stateful=True)

        self.linear = Dense(vocab_size, name="out")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if x.shape[0] is not None:
            self.batch_size = x.shape[0]

        x = self.embedding(x)
        x = self.rnn(x)
        x = self.linear(x)

        return x

    def reset_state(self) -> None:
        """Restore the cell's identity-based initial memory before starting a new sequence.

        Leave unbuilt recurrent state unchanged and assign initial hidden and memory states in place.

        """

        if self.rnn.states is None:
            return

        initial_states = self.cell.get_initial_state(batch_size=tf.shape(self.rnn.states[0])[0])
        for state, initial_state in zip(self.rnn.states, initial_states):
            state.assign(initial_state)
