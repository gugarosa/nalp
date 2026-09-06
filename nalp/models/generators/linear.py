# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Linear generator."""

import tensorflow as tf
from tensorflow.keras.layers import Dense

from nalp.core.model import Generator


class LinearGenerator(Generator):
    """Generate tanh outputs from noise with dense layers."""

    def __init__(
        self,
        input_shape: tuple[int, ...] = (784,),
        noise_dim: int = 100,
        n_samplings: int = 3,
        alpha: float = 0.01,
    ) -> None:
        """Initialize hidden dense layers and the output projection.

        Calls preserve leading noise dimensions and return tanh values with input_shape[0] final features.
        The training flag is accepted without changing the computation.

        Args:
            input_shape: Output shape whose first dimension sets the final feature count.
            noise_dim: Number of noise dimensions.
            n_samplings: Number of upsamplings to perform.
            alpha: Negative slope of the LeakyReLU activation.

        """

        super().__init__(name="G_linear")

        self.alpha = alpha
        self.noise_dim = noise_dim

        self.linear = [Dense(128 * (i + 1), name=f"linear_{i}") for i in range(n_samplings)]

        self.out = Dense(input_shape[0], activation="tanh", name="out")

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for layer in self.linear:
            x = tf.nn.leaky_relu(layer(x), self.alpha)

        x = self.out(x)

        return x
