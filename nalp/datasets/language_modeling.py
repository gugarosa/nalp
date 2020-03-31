from tensorflow import data

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class LanguageModelingDataset(Dataset):
    """A LanguageModelingDataset class is responsible for creating a dataset that predicts the next timestep (t+1)
    given a timestep (t).

    """

    def __init__(self, encoded_tokens, max_length=1, batch_size=64, shuffle=True):
        """Initialization method.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            max_length (int): Maximum sequences' length.
            batch_size (int): Size of batches.
            shuffle (bool): Whether batches should be shuffled or not.

        """

        logger.info('Overriding class: Dataset -> LanguageModelingDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(LanguageModelingDataset, self).__init__(encoded_tokens, shuffle)

        # Creating the sequences
        sequences = self._create_sequences(encoded_tokens, max_length)

        # Mapping the sequences to input and targets
        mapped_sequences = sequences.map(self._create_input_target)

        # Building up the dataset class
        self._build(mapped_sequences, batch_size)

        # Debugging some important information
        logger.debug(
            f'Batches: {data.experimental.cardinality(self.batches)} | Batch size: {batch_size} | Shuffle: {shuffle}.')

        logger.info('Class overrided.')

    def _create_sequences(self, encoded_tokens, max_length):
        """Creates sequences of the desired length.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            max_length (int): Maximum sequences' length.

        Returns:
            A tensor of maximum length sequences.

        """

        logger.debug(f'Creating sequences ...')

        # Creating tensor slices from the encoded tokens
        slices = data.Dataset.from_tensor_slices(encoded_tokens)

        # Creating the sequences
        sequences = slices.batch(max_length + 1, drop_remainder=True)

        logger.debug(f'Maximum length: {max_length}.')

        return sequences

    def _create_input_target(self, sequence):
        """Creates input (t) and targets (t+1) using the next timestep approach.

        Args:
            sequence (tensor): A tensor holding the sequence to be mapped.

        Returns:
            Input and target tensors.

        """

        # Maps the sequence to the input
        input = sequence[:-1]

        # Maps the sequence to the target
        target = sequence[1:]

        return input, target

    def _build(self, mapped_sequences, batch_size):
        """Builds the batches based on the mapped sequences.

        Args:
            mapped_sequences (tf.Tensor): A tensor of mapped sequences.
            batch_size (int): Size of batches.

        """

        # Checks if data should be shuffled
        if self.shuffle:
            # Creating the dataset from shuffled and batched data
            self.batches = mapped_sequences.shuffle(
                c.BUFFER_SIZE).batch(batch_size, drop_remainder=True)

        # If should not be shuffled
        else:
            # Creating the dataset from batched data
            self.batches = mapped_sequences.batch(
                batch_size, drop_remainder=True)
