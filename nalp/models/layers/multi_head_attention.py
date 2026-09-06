# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Multi-head attention layer."""

from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


def scaled_dot_product_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return scaled dot-product attention and its weights.

    The mask must broadcast to the attention logits, with nonzero entries excluding keys.
    Fully masked rows return zero weights and zero outputs.

    Args:
        q: Queries shaped (..., query_length, depth).
        k: Keys shaped (..., key_length, depth).
        v: Values shaped (..., key_length, value_depth).
        mask: Optional exclusion mask broadcastable to (..., query_length, key_length).

    Returns:
        Outputs shaped (..., query_length, value_depth) and weights shaped (..., query_length, key_length).

    """

    logits = tf.matmul(q, k, transpose_b=True)
    logits /= tf.math.sqrt(tf.cast(tf.shape(k)[-1], logits.dtype))

    if mask is not None:
        mask = tf.cast(mask, tf.bool)
        logits = tf.where(mask, tf.cast(float("-inf"), logits.dtype), logits)
        # Zero fully masked rows around softmax to avoid undefined all-negative-infinity normalization
        logits = tf.where(tf.reduce_all(mask, axis=-1, keepdims=True), 0.0, logits)

    weights = tf.nn.softmax(logits, -1)
    if mask is not None:
        weights = tf.where(mask, 0.0, weights)

    return tf.matmul(weights, v), weights


class MultiHeadAttention(Layer):
    """Multi-head attention with NALP's original public API."""

    def __init__(self, n_features: int, n_heads: int, **kwargs) -> None:
        """Initialize query, key, value, and output projections.

        Calls accept batched query, key, and value sequences and return projected attention and per-head weights.
        Outputs have shape (batch_size, query_length, n_features), while weights have shape
        (batch_size, n_heads, query_length, key_length). Nonzero mask entries exclude keys and must broadcast
        to the weights. Fully masked rows have zero attention weights before the output projection.

        Reference: A. Vaswani et al. Attention is all you need. NeurIPS (2017).

        Args:
            n_features: Number of projected features shared evenly across attention heads.
            n_heads: Number of attention heads.
            **kwargs: Keras layer keyword arguments.

        Raises:
            ValueError: The feature count is not divisible by the number of heads.

        """

        super().__init__(**kwargs)

        if n_features % n_heads:
            raise ValueError(f"`n_features` must be divisible by n_heads, but got {n_features} for {n_heads} heads.")

        self.n_features = n_features
        self.n_heads = n_heads
        self.depth = n_features // n_heads

        self.w_q = Dense(n_features)
        self.w_k = Dense(n_features)
        self.w_v = Dense(n_features)

        self.out = Dense(n_features)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        attention, weights = scaled_dot_product_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (tf.shape(attention)[0], -1, self.n_features))

        return self.out(attention), weights

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "n_features": self.n_features,
            "n_heads": self.n_heads,
        }
