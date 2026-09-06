# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Linear discriminator."""

import tensorflow as tf
from tensorflow.keras.layers import Dense

from nalp.core.model import Discriminator


class LinearDiscriminator(Discriminator):
    """Discriminate samples with dense layers and scalar logits."""

    def __init__(self, n_samplings: int = 3, alpha: float = 0.01) -> None:
        """Initialize hidden dense layers and the scalar-logit projection.

        Calls preserve leading input dimensions and replace the final feature dimension with one logit.
        The training flag is accepted without changing the computation.

        Args:
            n_samplings: Number of downsamplings to perform.
            alpha: Negative slope of the LeakyReLU activation.

        """

        super().__init__(name="D_linear")

        self.alpha = alpha

        self.linear = [Dense(128 * i, name=f"linear_{i}") for i in range(n_samplings, 0, -1)]

        self.out = Dense(1, name="out")

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for layer in self.linear:
            x = tf.nn.leaky_relu(layer(x), self.alpha)

        x = self.out(x)

        return x
