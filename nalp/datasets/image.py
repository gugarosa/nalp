"""Imaging dataset class.
"""

from tensorflow import data

import nalp.utils.logging as l
from nalp.core import Dataset

logger = l.get_logger(__name__)


class ImageDataset(Dataset):
    """An ImageDataset class is responsible for creating a dataset that encodes images for
    adversarial generation.

    """

    def __init__(self, images, batch_size=256, shape=None, normalize=True, shuffle=True):
        """Initialization method.

        Args:
            images (np.array): An array of images.
            batch_size (int): Size of batches.
            shape (tuple): A tuple containing the shape if the array should be forced to reshape.
            normalize (bool): Whether images should be normalized between -1 and 1.
            shuffle (bool): Whether batches should be shuffled or not.

        """

        logger.info('Overriding class: Dataset -> ImageDataset.')

        super(ImageDataset, self).__init__(shuffle)

        # Pre-process an array of images
        processed_images = self._preprocess(images, shape, normalize)

        # Building up the dataset class
        self._build(processed_images, batch_size)

        logger.debug('Size: %s | Batch size: %d | Normalization: %s | Shuffle: %s.',
                     shape, batch_size, normalize, self.shuffle)
        logger.info('Class overrided.')

    def _preprocess(self, images, shape, normalize):
        """Pre-process an array of images by reshaping and normalizing, if necessary.

        Args:
            images (np.array): An array of images.
            shape (tuple): A tuple containing the shape if the array should be forced to reshape.
            normalize (bool): Whether images should be normalized between -1 and 1.

        Returns:
            Slices of pre-processed tensor-based images.

        """

        images = images.astype('float32')

        if shape:
            images = images.reshape(shape)

        if normalize:
            images = (images - 127.5) / 127.5

        # Slices the arrays into tensors
        images = data.Dataset.from_tensor_slices(images)

        return images
