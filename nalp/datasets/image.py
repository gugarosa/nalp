from tensorflow import data

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core.dataset import Dataset

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

        # Pre-process an array of images
        processed_images = self._preprocess(images, shape, normalize)

        # Overrides its parent class with any custom arguments if needed
        super(ImageDataset, self).__init__(processed_images, shuffle)

        # Building up the dataset class
        self._build(processed_images, batch_size)

        # Debugging some important information
        logger.debug(
            f'Size: {shape} | Batch size: {batch_size} | Normalization: {normalize} | Shuffle: {shuffle}.')

        logger.info('Class overrided.')

    def _preprocess(self, images, shape, normalize):
        """Pre-process an array of images by reshaping and normalizing, if necessary.

        Args:
            images (np.array): An array of images.
            shape (tuple): A tuple containing the shape if the array should be forced to reshape.
            normalize (bool): Whether images should be normalized between -1 and 1.

        Returns:
            An array of pre-processed images.

        """

        # If a shape is supplied
        if shape:
            # Reshapes the array and make sure that it is float typed
            images = images.reshape(shape).astype('float32')

        # If no shape is supplied
        else:
            # Just make sure that the array is float typed
            images = images.astype('float32')

        # If images should be normalized
        if normalize:
            # Normalize the images between -1 and 1
            images = (images - 127.5) / 127.5

        return images

    def _build(self, processed_images, batch_size):
        """Builds the batches based on the pre-processed images.

        Args:
            processed_images (np.array): An array of pre-processed images.
            batch_size (int): Size of batches.

        """

        # Checks if data should be shuffled
        if self.shuffle:
            # Creating the dataset from shuffled and batched data
            self.batches = data.Dataset.from_tensor_slices(
                processed_images).shuffle(c.BUFFER_SIZE).batch(batch_size)

        # If should not be shuffled
        else:
            # Creating the dataset from batched data
            self.batches = data.Dataset.from_tensor_slices(
                processed_images).batch(batch_size)
