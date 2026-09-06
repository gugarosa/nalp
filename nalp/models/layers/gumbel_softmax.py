# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Gumbel-Softmax layer."""

from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Layer

import nalp.utils.constants as c


def gumbel_distribution(input_shape: tuple[int, ...]) -> tf.Tensor:
    """Sample float32 Gumbel noise with epsilon-stabilized inverse-transform sampling.

    Args:
        input_shape: Shape of tensor to be sampled.

    Returns:
        A float32 Gumbel noise tensor with the requested shape.

    """

    uniform_dist = tf.random.uniform(input_shape, 0, 1)
    gumbel_dist = -1 * tf.math.log(-1 * tf.math.log(uniform_dist + c.EPSILON) + c.EPSILON)

    return gumbel_dist


class GumbelSoftmax(Layer):
    """Relax categorical logits with Gumbel-Softmax sampling."""

    def __init__(self, axis: int = -1, **kwargs) -> None:
        """Initialize the axis used for relaxed probabilities and token selection.

        Calls add float32 Gumbel noise to input logits, divide by tau, and return (probabilities, token_ids).
        Probabilities retain the input shape. Token IDs are int32 argmax values with the selected axis removed
        and gradients stopped. Layer configuration retains the softmax axis.

        Reference: E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax.
        Preprint arXiv:1611.01144 (2016).

        Args:
            axis: Axis to perform the softmax operation.
            **kwargs: Keras layer keyword arguments.

        """

        super().__init__(**kwargs)

        self.axis = axis

    def call(self, inputs: tf.Tensor, tau: float) -> tuple[tf.Tensor, tf.Tensor]:
        x = inputs + gumbel_distribution(tf.shape(inputs))
        x = tf.nn.softmax(x / tau, self.axis)

        y = tf.stop_gradient(tf.argmax(x, self.axis, tf.int32))

        return x, y

    def get_config(self) -> dict[str, Any]:
        config = {"axis": self.axis}

        return {**super().get_config(), **config}
