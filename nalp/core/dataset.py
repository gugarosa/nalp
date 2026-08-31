"""Dataset-related class.
"""

import tensorflow as tf

import nalp.utils.constants as c


class Dataset:
    """A Dataset class is responsible for receiving encoded tokens and
    persisting data that will be feed as an input to the networks.

    """

    def __init__(self, shuffle: bool = True) -> None:
        """Initialization method.

        Args:
            shuffle: Whether batches should be shuffled or not.

        """

        self.shuffle = shuffle
        self.batches: tf.data.Dataset | None = None

    def _build(self, sliced_data: tf.Tensor, batch_size: int) -> None:
        """Builds the batches based on the pre-processed images.

        Args:
            sliced_data: Slices of tensor-based data.
            batch_size: Size of batches.

        """

        if self.shuffle:
            sliced_data = sliced_data.shuffle(c.BUFFER_SIZE)

        self.batches = sliced_data.batch(batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE
        )
