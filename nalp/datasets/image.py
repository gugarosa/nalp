"""Imaging dataset class."""

import numpy as np
import tensorflow as tf

from nalp.core import Dataset


class ImageDataset(Dataset):
    """An ImageDataset class is responsible for creating a dataset that encodes images for
    adversarial generation.

    """

    def __init__(
        self,
        images: np.ndarray,
        batch_size: int = 256,
        shape: tuple[int, int] | None = None,
        normalize: bool = True,
        shuffle: bool = True,
    ) -> None:
        """Initialization method.

        Args:
            images: An array of images.
            batch_size: Size of batches.
            shape: A tuple containing the shape if the array should be forced to reshape.
            normalize: Whether images should be normalized between -1 and 1.
            shuffle: Whether batches should be shuffled or not.

        """

        super().__init__(shuffle)

        processed_images = self._preprocess(images, shape, normalize)

        self._build(processed_images, batch_size)

    def _preprocess(
        self, images: np.ndarray, shape: tuple[int, int] | None, normalize: bool
    ) -> tf.data.Dataset:
        """Pre-process an array of images by reshaping and normalizing, if necessary.

        Args:
            images: An array of images.
            shape: A tuple containing the shape if the array should be forced to reshape.
            normalize: Whether images should be normalized between -1 and 1.

        Returns:
            (tf.data.Dataset): Slices of pre-processed tensor-based images.

        """

        images = images.astype("float32")

        if shape:
            images = images.reshape(shape)

        if normalize:
            images = (images - 127.5) / 127.5

        return tf.data.Dataset.from_tensor_slices(images)
