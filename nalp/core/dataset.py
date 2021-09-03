"""Dataset-related class.
"""

from tensorflow import data

import nalp.utils.constants as c


class Dataset:
    """A Dataset class is responsible for receiving encoded tokens and
    persisting data that will be feed as an input to the networks.

    """

    def __init__(self, shuffle=True):
        """Initialization method.

        Args:
            shuffle (bool): Whether batches should be shuffled or not.

        """

        # Creating a property to whether data should be shuffled or not
        self.shuffle = shuffle

    @property
    def shuffle(self):
        """bool: Whether data should be shuffled or not.

        """

        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    @property
    def batches(self):
        """tf.data.Dataset: An instance of tensorflow's dataset batches.

        """

        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches

    def _build(self, sliced_data, batch_size):
        """Builds the batches based on the pre-processed images.

        Args:
            sliced_data (tf.tensor): Slices of tensor-based data.
            batch_size (int): Size of batches.

        """

        if self.shuffle:
            sliced_data = sliced_data.shuffle(c.BUFFER_SIZE)

        # Transforms the sequences into batches
        self.batches = (
            sliced_data
            .batch(batch_size, drop_remainder=True)
            .prefetch(data.experimental.AUTOTUNE))
