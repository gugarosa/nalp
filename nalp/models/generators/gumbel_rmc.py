# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gumbel Relational Memory Core generator."""

import tensorflow as tf

from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators._gumbel import GumbelGeneratorMixin
from nalp.models.generators.rmc import RMCGenerator
from nalp.models.layers.gumbel_softmax import GumbelSoftmax


class GumbelRMCGenerator(GumbelGeneratorMixin, RMCGenerator):
    """Generate logits, relaxed probabilities, and token IDs with Gumbel relational memory."""

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
        tau: float = 5,
    ) -> None:
        """Initialize stateful relational memory with a Gumbel-Softmax output.

        Calls accept integer IDs shaped (batch_size, length) and return (logits, probabilities, token_ids).
        Logits and probabilities have shape (batch_size, length, vocab_size), while int32 token IDs omit the last axis.
        Recurrent state persists until reset_state restores identity-based memory. The mutable temperature is a
        nontrainable float32 resource outside Keras weight lists and remains visible to traced training.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of attention and feed-forward refinement blocks.
            n_layers: Number of layers per feed-forward network.
            tau: Gumbel-Softmax temperature parameter.

        Raises:
            ValueError: The temperature is non-finite or not positive.

        """

        super().__init__(
            encoder,
            vocab_size,
            embedding_size,
            n_slots,
            n_heads,
            head_size,
            n_blocks,
            n_layers,
        )

        self.tau = tau

        self.gumbel = GumbelSoftmax(name="gumbel")

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = super().call(x)

        x_g, y_g = self.gumbel(x, tau=self._tau_tensor)

        return x, x_g, y_g
