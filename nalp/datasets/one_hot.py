import nalp.utils.logging as l
import numpy as np
from nalp.core.dataset import Dataset

logger = l.get_logger(__name__)


class OneHot(Dataset):
    """An OneHot encoding Dataset child. It is responsible for receiving and preparing data
    in a one-hot format. This data serves as a basis for predicting t+1 timesteps.

    Properties:
        max_length (int): The maximum length of the encoding.
        X (np.array): Input samples already in one-hot encoding format.
        Y (np.array): Target samples already in one-hot encoding format.

    Methods:
        encode_tokens(tokens_idx, max_length, vocab_size): Encodes indexed tokens into one-hot format.

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

        # Calls encode function and creates a (X, Y) for further using
        self._X, self._Y = self.encode_tokens(
            self._tokens_idx, max_length, self._vocab_size)

        # Logging some important information
        logger.debug(
            f'X: {self._X.shape} | Y: {self._Y.shape}.')

        logger.info('Class overrided.')

    @property
    def max_length(self):
        """The maximum length of the encoding.

        """

        return self._max_length

    @property
    def X(self):
        """Input samples already in one-hot encoding format.

        """

        return self._X

    @property
    def Y(self):
        """Target samples already in one-hot encoding format.

        """

        return self._Y

    def encode_tokens(self, tokens_idx, max_length, vocab_size):
        """Encodes indexed tokens into one-hot format.

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
        X = np.zeros((len(inputs), max_length, vocab_size), dtype=np.int32)
        Y = np.zeros((len(inputs), vocab_size), dtype=np.int32)

        # Iterates through all inputs
        for i, input in enumerate(inputs):
            # For each input, iterate through all tokens
            for t, token in enumerate(input):
                # If there is a token on X, mark as true
                X[i, t, token] = 1
            # If there is a token on Y, mark as true
            Y[i, targets[i]] = 1

        return X, Y
