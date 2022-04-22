"""Imaging dataset class.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from nalp.core import Dataset
from nalp.utils import logging

logger = logging.get_logger(__name__)


class ImageDataset(Dataset):
    """An ImageDataset class is responsible for creating a dataset that encodes images for
    adversarial generation.

    """

    def __init__(
        self,
        images: np.array,
        batch_size: Optional[int] = 256,
        shape: Optional[Tuple[int, int]] = None,
        normalize: Optional[bool] = True,
        shuffle: Optional[bool] = True,
    ) -> None:
        """Initialization method.

        Args:
            images: An array of images.
            batch_size: Size of batches.
            shape: A tuple containing the shape if the array should be forced to reshape.
            normalize: Whether images should be normalized between -1 and 1.
            shuffle: Whether batches should be shuffled or not.

        """

        logger.info("Overriding class: Dataset -> ImageDataset.")

        super(ImageDataset, self).__init__(shuffle)

        # Pre-process an array of images
        processed_images = self._preprocess(images, shape, normalize)

        # Building up the dataset class
        self._build(processed_images, batch_size)

        logger.debug(
            "Size: %s | Batch size: %d | Normalization: %s | Shuffle: %s.",
            shape,
            batch_size,
            normalize,
            self.shuffle,
        )
        logger.info("Class overrided.")

    def _preprocess(
        self, images: np.array, shape: Tuple[int, int], normalize: bool
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

        # Slices the arrays into tensors
        images = tf.data.Dataset.from_tensor_slices(images)

        return images
