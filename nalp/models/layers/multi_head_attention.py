"""Multi-head attention layer."""

from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

import nalp.utils.constants as c


def scaled_dot_product_attention(
    q: tf.Tensor,
    k: tf.Tensor,
    v: tf.Tensor,
    mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return scaled dot-product attention and its weights."""

    logits = tf.matmul(q, k, transpose_b=True)
    logits /= tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))

    if mask is not None:
        logits += mask * c.EPSILON

    weights = tf.nn.softmax(logits, -1)
    return tf.matmul(weights, v), weights


class MultiHeadAttention(Layer):
    """Multi-head attention with NALP's original public API."""

    def __init__(self, n_features: int, n_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if n_features % n_heads:
            raise ValueError("n_features must be divisible by n_heads.")

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
        """Apply attention to query, key, and value tensors."""

        q = self._split_heads(self.w_q(q))
        k = self._split_heads(self.w_k(k))
        v = self._split_heads(self.w_v(v))

        attention, weights = scaled_dot_product_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(
            attention, (tf.shape(attention)[0], -1, self.n_features)
        )
        return self.out(attention), weights

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration."""

        return {
            **super().get_config(),
            "n_features": self.n_features,
            "n_heads": self.n_heads,
        }
