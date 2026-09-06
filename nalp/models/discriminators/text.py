# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Text-based discriminator."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool1D

from nalp.core.model import Discriminator


class TextDiscriminator(Discriminator):
    """Discriminate vocabulary distributions with convolutional highway features."""

    def __init__(
        self,
        max_length: int = 1,
        embedding_size: int = 32,
        n_filters: tuple[int, ...] = (64,),
        filters_size: tuple[int, ...] = (1,),
        dropout_rate: float = 0.25,
    ) -> None:
        """Initialize the dense embedding, convolutional pooling, and highway projection.

        Calls accept distributions shaped (batch_size, max_length, vocab_size) and return unnormalized features
        shaped (batch_size, 1, sum(n_filters)), not two-class logits. Dropout follows the training flag.

        Args:
            max_length: Maximum length of the sequences.
            embedding_size: The size of the embedding layer.
            n_filters: Number of filters to be applied.
            filters_size: Size of filters to be applied.
            dropout_rate: Dropout activation rate.

        """

        super().__init__(name="D_text")

        self.embedding = Dense(embedding_size, name="embedding")

        self.conv = [
            Conv2D(
                n,
                (k, embedding_size),
                strides=(1, 1),
                padding="valid",
                name=f"conv_{k}",
            )
            for n, k in zip(n_filters, filters_size)
        ]

        self.pool = [MaxPool1D(max_length - k + 1, 1, name=f"pool_{k}") for k in filters_size]

        self.highway = Dense(sum(n_filters), name="highway")

        self.drop = Dropout(dropout_rate, name="drop")

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        convs = [tf.squeeze(tf.nn.relu(conv(x)), 2) for conv in self.conv]
        pools = [pool(conv) for pool, conv in zip(self.pool, convs)]

        x = tf.concat(pools, 2)
        hw = self.highway(x)
        x = tf.math.sigmoid(hw) * tf.nn.relu(hw) + (1 - tf.math.sigmoid(hw)) * x

        x = self.drop(x, training=training)

        return x
