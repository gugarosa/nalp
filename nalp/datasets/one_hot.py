import nalp.utils.logging as l
import numpy as np
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class OneHot(Dataset):
    """An OneHot encoding Dataset child. It is responsible for receiving and preparing data
    in a one-hot format. This data serves as a basis for predicting t+1 timesteps.

    """

    def __init__(self, tokens, max_length=1):
        """Initizaliation method.

        Args:
            tokens (list): A list holding tokenized words or characters.
            max_length (int): The maximum length of the encoding.

        """

        logger.info('Overriding class: Dataset -> OneHot.')

        # Overrides its parent class with any custom arguments if needed
        super(OneHot, self).__init__(tokens)

        # We need to create a property holding the max length of the encoding
        self._max_length = max_length

        # Calls creating samples method to populate (X, Y) for further using
        self.X, self.Y = self.create_samples(
            self.tokens_idx, max_length, self.vocab_size)

        # Logging some important information
        logger.debug(
            f'X: {self.X.shape} | Y: {self.Y.shape}.')

        logger.info('Class overrided.')

    @property
    def max_length(self):
        """int: The maximum length of the encoding.

        """

        return self._max_length

    def one_hot_encode(self, token_idx, vocab_size):
        """Encodes an indexated token into an one-hot encoding.

        Args:
            token_idx (int): The index of the token to be encoded.
            vocab_size (int): The size of the vocabulary.

        Returns:
            A one-hot encoded array.

        """

        # Creating array to hold encoded data
        encoded_data = np.zeros((vocab_size), dtype=np.int32)

        # Marking as true where tokens exists
        encoded_data[token_idx] = 1

        return encoded_data


    def create_samples(self, tokens_idx, max_length, vocab_size):
        """Creates inputs and targets samples based in a one-hot encoding.
        We are predicting t+1 timesteps for each char or word.

        Args:
            tokens_idx (np.array): A numpy array holding the indexed tokens.
            max_length (int): The maximum length of the encoding.
            vocab_size (int): The size of the vocabulary.

        Returns:
            X and Y one-hot encoded samples.

        """

        # Creates empty lists for further appending
        inputs = []
        targets = []

        # Iterates through all possible inputs combinations
        for i in range(0, len(tokens_idx)-max_length):
            # Appends to inputs and targets lists one timestep at a time
            inputs.append(tokens_idx[i:i+max_length])
            targets.append(tokens_idx[i+max_length])

        # Creates empty numpy boolean arrays for holding X and Y
        X = np.zeros((len(inputs), max_length, vocab_size), dtype=np.float32)
        Y = np.zeros((len(inputs), vocab_size), dtype=np.float32)

        # Iterates through all inputs
        for i, input in enumerate(inputs):
            # For each input, iterate through all tokens
            for t, token in enumerate(input):
                # If there is a token on X, encode it
                X[i, t] = self.one_hot_encode(token, vocab_size)
            # If there is a token on Y, encode it
            Y[i] = self.one_hot_encode(targets[i], vocab_size)

        return X, Y
