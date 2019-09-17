import nalp.utils.logging as l
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class NextDataset(Dataset):
    """A NextDataset class is responsible for creating a dataset that predicts the next timestep (t+1)
    given a timestep (t).

    """

    def __init__(self, encoded_tokens, max_length=1, batch_size=64):
        """Initialization method.

        Args:
            encoded_tokens (np.array): An array of encoded tokens.
            max_length (int): Maximum sequences' length.
            batch_size (int): Size of batches.

        """

        logger.info('Overriding class: Dataset -> NextDataset.')

        # Overrides its parent class with any custom arguments if needed
        super(NextDataset, self).__init__(encoded_tokens, max_length)

        # Creating the sequences
        sequences = self._create_sequences()

        # Mapping the sequences to input and targets
        map_sequences = sequences.map(self._create_input_target)

        logger.debug(
            f'Creating input and target batches of size: {batch_size}.')

        # Actually creating the desired amount of batches
        self.batches = map_sequences.shuffle(
            10000).batch(batch_size, drop_remainder=True)

        logger.info('Class overrided.')

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
