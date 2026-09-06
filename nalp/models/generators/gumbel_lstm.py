# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gumbel Long Short-Term Memory generator."""

import tensorflow as tf

from nalp.encoders.integer import IntegerEncoder
from nalp.models.generators._gumbel import GumbelGeneratorMixin
from nalp.models.generators.lstm import LSTMGenerator
from nalp.models.layers.gumbel_softmax import GumbelSoftmax


class GumbelLSTMGenerator(GumbelGeneratorMixin, LSTMGenerator):
    """Generate logits, relaxed probabilities, and token IDs with a Gumbel LSTM."""

    def __init__(
        self,
        encoder: IntegerEncoder | None = None,
        vocab_size: int = 1,
        embedding_size: int = 32,
        hidden_size: int = 64,
        tau: float = 5.0,
    ) -> None:
        """Initialize a stateful LSTM with a Gumbel-Softmax output.

        Calls accept integer IDs shaped (batch_size, length) and return (logits, probabilities, token_ids).
        Logits and probabilities have shape (batch_size, length, vocab_size), while int32 token IDs omit the last axis.
        Recurrent state persists until reset_state is invoked. The mutable temperature is a nontrainable float32
        resource outside Keras weight lists, so assignments remain visible to traced training without changing weights.

        Args:
            encoder: An index to vocabulary encoder.
            vocab_size: The size of the vocabulary.
            embedding_size: The size of the embedding layer.
            hidden_size: The amount of hidden neurons.
            tau: Gumbel-Softmax temperature parameter.

        Raises:
            ValueError: The temperature is non-finite or not positive.

        """

        super().__init__(encoder, vocab_size, embedding_size, hidden_size)

        self.tau = tau

        self.gumbel = GumbelSoftmax(name="gumbel")

    def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = super().call(x)

        x_g, y_g = self.gumbel(x, tau=self._tau_tensor)

        return x, x_g, y_g
