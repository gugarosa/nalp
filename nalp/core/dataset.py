# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Dataset-related class."""

import tensorflow as tf

import nalp.utils.constants as c


class Dataset:
    """Hold a TensorFlow batching pipeline and its shuffle configuration."""

    def __init__(self, shuffle: bool = True) -> None:
        """Initialize an unbuilt batching pipeline.

        Args:
            shuffle: Whether to shuffle individual samples before batching.

        """

        self.shuffle = shuffle
        self.batches: tf.data.Dataset | None = None

    def _build(self, sliced_data: tf.data.Dataset, batch_size: int) -> None:
        if self.shuffle:
            sliced_data = sliced_data.shuffle(c.BUFFER_SIZE)

        self.batches = sliced_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
