# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Imaging dataset class."""

import numpy as np
import tensorflow as tf

from nalp.core.dataset import Dataset


class ImageDataset(Dataset):
    """Prepare float32 image batches for adversarial training."""

    def __init__(
        self,
        images: np.ndarray,
        batch_size: int = 256,
        shape: tuple[int, ...] | None = None,
        normalize: bool = True,
        shuffle: bool = True,
    ) -> None:
        """Convert images and create a batched TensorFlow dataset.

        Copy input values to float32 before optional reshaping and normalization.
        Incomplete final batches are discarded, and the original array is not modified.

        Args:
            images: Image values in the range 0 through 255 when normalization is enabled.
            batch_size: Number of images in each complete batch.
            shape: Optional reshape target including the leading sample dimension.
            normalize: Whether to map pixel values from the 0-to-255 range into the -1-to-1 range.
            shuffle: Whether to shuffle individual images before batching.

        Raises:
            ValueError: The requested shape is incompatible with the input array.

        """

        super().__init__(shuffle)

        processed_images = self._preprocess(images, shape, normalize)

        self._build(processed_images, batch_size)

    def _preprocess(self, images: np.ndarray, shape: tuple[int, ...] | None, normalize: bool) -> tf.data.Dataset:
        images = images.astype("float32")

        if shape:
            images = images.reshape(shape)

        if normalize:
            images = (images - 127.5) / 127.5

        return tf.data.Dataset.from_tensor_slices(images)
