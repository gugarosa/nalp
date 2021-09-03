"""Multi-Head Attention layer.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

import nalp.utils.constants as c


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights, such that q, k, v must have matching
    leading dimensions, and k, v must have matching penultimate dimension.

    Args:
        q (tf.tensor): Query tensor.
        k (tf.tensor): Key tensor.
        v (tf.tensor): Value tensor.
        mask (tf.tensor): Mask to be applied.

    Returns:
        An attention-based output tensor and its attention weights.

    """

    # Performs the multiplication between `q` and `k`
    qk = tf.matmul(q, k, transpose_b=True)

    # Casts the `qk` tensor to a float type
    scaled_qk = tf.cast(tf.shape(k)[-1], tf.float32)

    # Scales the tensor
    scaled_attention_logits = qk / tf.math.sqrt(scaled_qk)

    if mask is not None:
        # Adds the mask to the scaled attention logits
        scaled_attention_logits += (mask * c.EPSILON)

    # Calculates the attention weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, -1)

    # Calculates the output tensor
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    """A MultiHeadAttention class is the one in charge of a Multi-Head Attention layer implementation.

    References:
        A.Vaswani, et al. Attention is all you need.
        Advances in neural information processing systems (2017).

    """

    def __init__(self, n_features, n_heads, **kwargs):
        """Initialization method.

        Args:
            n_features (int): Number of input features.
            n_heads (int): Number of attention heads.

        """

        super(MultiHeadAttention, self).__init__(**kwargs)

        # Defining a property to hold the number of input features
        self.n_features = n_features

        # Defining a property to hold the number of heads
        self.n_heads = n_heads

        # Checking if the number of features is divisible by the number of heads
        assert self.n_features % self.n_heads == 0

        # Calculating the depth of the heads
        self.depth = n_features // self.n_heads

        # Creates a linear layer for holding the `q` matrix
        self.w_q = Dense(n_features)

        # Creates a linear layer for holding the `k` matrix
        self.w_k = Dense(n_features)

        # Creates a linear layer for holding the `v` matrix
        self.w_v = Dense(n_features)

        # Creating the final linear layer
        self.out = Dense(n_features)

    def _split_heads(self, x):
        """Split the last tensor dimension into (n_heads, depth) and transposes its result.

        Args:
            x (tf.tensor): Tensor to be splitted and transposed.

        Returns:
            Splitted and transposed tensor into shape equal to (batch_size, n_heads, length, depth).

        """

        # Reshapes the tensor according to desired shape
        x = tf.reshape(x, (x.shape[0], -1, self.n_heads, self.depth))

        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        """Method that holds vital information whenever this class is called.

        Args:
            q (tf.tensor): Query tensor.
            k (tf.tensor): Key tensor.
            v (tf.tensor): Value tensor.
            mask (tf.tensor): Mask to be applied.

        Returns:
            An attention-based output tensor and its attention weights.

        """

        # Passes `q`, `k` and `v` down its linear layer
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Splits `q`, `k` and `v` into multiple heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Pefrosm the product for calculating the attention-based output
        attn, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Transposes the tensor into (batch_size, length, n_heads, depth)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])

        # Concatenates the multiple heads into a single tensor
        concat_attn = tf.reshape(attn, (attn.shape[0], -1, self.n_features))

        # Calculates the output vector
        output = self.out(concat_attn)

        return output, attn_weights

    def get_config(self):
        """Gets the configuration of the layer for further serialization.

        """

        config = {
            'n_features': self.n_features,
            'n_heads': self.n_heads,
            'depth': self.depth
        }
        base_config = super(MultiHeadAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
