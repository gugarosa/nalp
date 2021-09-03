"""Language modeling dataset class.
"""

from tensorflow import data

import nalp.utils.logging as l
from nalp.core import Dataset

logger = l.get_logger(__name__)


class LanguageModelingDataset(Dataset):
    """A LanguageModelingDataset class is responsible for creating a dataset
    that predicts the next timestep (t+1) given a timestep (t).

    """

    def __init__(self, encoded_tokens, max_contiguous_pad_length=1, batch_size=64, shuffle=True):
        """Initialization method.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            max_contiguous_pad_length (int): Maximum length to pad contiguous text.
            batch_size (int): Size of batches.
            shuffle (bool): Whether batches should be shuffled or not.

        """

        logger.info('Overriding class: Dataset -> LanguageModelingDataset.')

        super(LanguageModelingDataset, self).__init__(shuffle)

        # Creates the sequences and maps their inputs and targets
        sequences = self._create_sequences(encoded_tokens, encoded_tokens.ndim, max_contiguous_pad_length)
        mapped_sequences = sequences.map(self._create_input_target)

        # Builds up the dataset class
        self._build(mapped_sequences, batch_size)
        
        logger.debug('Batch size: %d | Shuffle: %s.', batch_size, self.shuffle)
        logger.info('Class overrided.')

    def _create_sequences(self, encoded_tokens, rank, max_contiguous_pad_length):
        """Creates sequences of the desired length.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            rank (int): Number of array dimensions (rank).
            max_contiguous_pad_length (int): Maximum sequences' length.

        Returns:
            Slices of tensor-based sequences.

        """

        # Slices the tensors into sequences
        sequences = data.Dataset.from_tensor_slices(encoded_tokens)

        # This means that is a contiguous sequence of tokens and needs to
        # be parsed into individual sequences
        if rank == 1:
            # Creates the sequences
            sequences = sequences.batch(max_contiguous_pad_length + 1, drop_remainder=True)

        return sequences

    def _create_input_target(self, sequence):
        """Creates input (t) and targets (t+1) using the next timestep approach.

        Args:
            sequence (tensor): A tensor holding the sequence to be mapped.

        Returns:
            Input and target tensors.

        """

        # Maps the sequence to input and target
        _input = sequence[:-1]
        target = sequence[1:]

        return _input, target
