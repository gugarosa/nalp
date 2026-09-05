import numpy as np
import pytest
import tensorflow as tf

from nalp.models.layers import MultiHeadAttention
from nalp.models.layers.multi_head_attention import scaled_dot_product_attention


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
@pytest.mark.parametrize("boolean_mask", [False, True])
def test_masked_keys_cannot_affect_attention(dtype, boolean_mask):
    mask = (
        tf.constant([[[False, True]]])
        if boolean_mask
        else tf.constant([[[0.0, 1.0]]], dtype)
    )
    outputs, weights = scaled_dot_product_attention(
        tf.ones((1, 1, 1), dtype),
        tf.ones((1, 2, 1), dtype),
        tf.constant([[[3.0], [100.0]]], dtype),
        mask,
    )

    assert outputs.dtype == weights.dtype == dtype
    np.testing.assert_allclose(outputs, [[[3.0]]])
    np.testing.assert_array_equal(weights, [[[1.0, 0.0]]])


def test_fully_masked_rows_are_zero_and_have_finite_gradients():
    query = tf.Variable([[[1.0], [2.0]]])
    with tf.GradientTape() as tape:
        outputs, weights = scaled_dot_product_attention(
            query,
            tf.ones((1, 2, 1)),
            tf.constant([[[3.0], [100.0]]]),
            tf.constant([[[1.0, 1.0], [0.0, 1.0]]]),
        )
        loss = tf.reduce_sum(outputs)

    np.testing.assert_array_equal(outputs, [[[0.0], [3.0]]])
    np.testing.assert_array_equal(weights, [[[0.0, 0.0], [1.0, 0.0]]])
    assert np.isfinite(tape.gradient(loss, query)).all()


def test_multi_head_layer_applies_broadcast_masks():
    layer = MultiHeadAttention(n_features=4, n_heads=2)
    inputs = tf.ones((2, 3, 4))
    _, weights = layer(inputs, inputs, inputs, mask=tf.constant([0.0, 0.0, 1.0]))

    np.testing.assert_array_equal(weights[:, :, :, 2], 0.0)
    np.testing.assert_allclose(tf.reduce_sum(weights, axis=-1), 1.0)
