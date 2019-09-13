import tensorflow as tf

import nalp.utils.logging as l

logger = l.get_logger(__name__)


class Dataset():
    """A Dataset class is responsible for receiving encoded tokens and
    creating data that will be feed as an input to the networks.

    """

    def __init__(self, encoded_tokens, max_length=1):
        """Initialization method.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            max_length (int): Maximum sequences' length.

        """

        # Creating a property to hold the encoded tokens
        self.encoded_tokens = encoded_tokens

        # We need to create a property holding the max length of the sequences
        self.max_length = max_length

    @property
    def encoded_tokens(self):
        """np.array: An numpy array holding the encoded tokens.

        """

        return self._encoded_tokens

    @encoded_tokens.setter
    def encoded_tokens(self, encoded_tokens):
        self._encoded_tokens = encoded_tokens

    @property
    def max_length(self):
        """int: The maximum length of the sequences.

        """

        return self._max_length

    @max_length.setter
    def max_length(self, max_length):
        self._max_length = max_length

    def _create_sequences(self):
        """Creates sequences of the desired length.

        Returns:
            A tensor of maximum length sequences.

        """

        logger.debug(
            f'Creating sequences of maximum length: {self.max_length} ...')

        # Creating tensor slices from the encoded tokens
        slices = tf.data.Dataset.from_tensor_slices(self.encoded_tokens)

        # Creating the sequences
        sequences = slices.batch(self.max_length+1, drop_remainder=True)

        return sequences
