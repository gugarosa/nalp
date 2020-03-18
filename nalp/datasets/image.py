from tensorflow import data

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class ImageDataset(Dataset):
    """An ImageDataset class is responsible for creating a dataset that encodes images for
    adversarial generation.

    """

    def __init__(self, images, batch_size=256, shape=None, normalize=True):
        """Initialization method.

        Args:
            images (np.array): An array of images.
            batch_size (int): Size of batches.
            normalize (bool): Whether images should be normalized between -1 and 1.

        """

        logger.info('Overriding class: Dataset -> ImageDataset.')

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

        # Overrides its parent class with any custom arguments if needed
        super(ImageDataset, self).__init__(images)

        # Creating the dataset from shuffled and batched data
        self.batches = data.Dataset.from_tensor_slices(images).shuffle(c.BUFFER_SIZE).batch(batch_size)

        # Debugging some important information
        logger.debug(
            f'Shape: {shape} | Batch size: {batch_size} | Normalization: {normalize}.')

        logger.info('Class overrided.')
