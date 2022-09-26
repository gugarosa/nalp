"""Multi-Head Attention layer.
"""

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

import nalp.utils.constants as c


def scaled_dot_product_attention(
    q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Calculate the attention weights, such that q, k, v must have matching
    leading dimensions, and k, v must have matching penultimate dimension.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        mask: Mask to be applied.

    Returns:
        (Tuple[tf.Tensor, tf.Tensor]): An attention-based output tensor and its attention weights.

    """

    qk = tf.matmul(q, k, transpose_b=True)

    scaled_qk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = qk / tf.math.sqrt(scaled_qk)

    if mask is not None:
        scaled_attention_logits += mask * c.EPSILON

    attention_weights = tf.nn.softmax(scaled_attention_logits, -1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    """A MultiHeadAttention class is the one in charge of a Multi-Head Attention layer implementation.

    References:
        A.Vaswani, et al. Attention is all you need.
        Advances in neural information processing systems (2017).

    """

    def __init__(self, n_features: int, n_heads: int, **kwargs) -> None:
        """Initialization method.

        Args:
            n_features: Number of input features.
            n_heads: Number of attention heads.

        """

        super(MultiHeadAttention, self).__init__(**kwargs)

        self.n_features = n_features
        self.n_heads = n_heads
        assert self.n_features % self.n_heads == 0

        self.depth = n_features // self.n_heads

        self.w_q = Dense(n_features)
        self.w_k = Dense(n_features)
        self.w_v = Dense(n_features)

        self.out = Dense(n_features)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """Split the last tensor dimension into (n_heads, depth) and transposes its result.

        Args:
            x: Tensor to be splitted and transposed.

        Returns:
            (tf.Tensor): Splitted and transposed tensor into shape equal to (batch_size, n_heads, length, depth).

        """

        x = tf.reshape(x, (x.shape[0], -1, self.n_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Method that holds vital information whenever this class is called.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Mask to be applied.

        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): An attention-based output tensor and its attention weights.

        """

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])

        concat_attn = tf.reshape(attn, (attn.shape[0], -1, self.n_features))

        output = self.out(concat_attn)

        return output, attn_weights

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.

        Returns:
            (Dict[str, Any]): Configuration dictionary.

        """

        config = {
            "n_features": self.n_features,
            "n_heads": self.n_heads,
            "depth": self.depth,
        }
        base_config = super(MultiHeadAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
