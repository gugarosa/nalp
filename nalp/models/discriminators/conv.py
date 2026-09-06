# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Convolutional discriminator."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout

from nalp.core.model import Discriminator


class ConvDiscriminator(Discriminator):
    """Discriminate channels-last images with convolutional downsampling."""

    def __init__(
        self,
        n_samplings: int = 3,
        alpha: float = 0.3,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize convolution, dropout, and scalar-logit layers.

        Calls accept channels-last images and return one unnormalized logit per downsampled spatial position.
        The batch and remaining spatial axes are retained, and dropout follows the training flag.

        Args:
            n_samplings: Number of downsamplings to perform.
            alpha: Negative slope of the LeakyReLU activation.
            dropout_rate: Dropout activation rate.

        """

        super().__init__(name="D_conv")

        self.alpha = alpha

        self.conv = [
            Conv2D(64 * (i + 1), (5, 5), strides=(2, 2), padding="same", name=f"conv_{i}") for i in range(n_samplings)
        ]

        self.drop = [Dropout(dropout_rate, name=f"drop_{i}") for i in range(n_samplings)]

        self.out = Dense(1, name="out")

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for c, d in zip(self.conv, self.drop):
            x = d(tf.nn.leaky_relu(c(x), self.alpha), training=training)

        x = self.out(x)

        return x
