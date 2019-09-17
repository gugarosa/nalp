import numpy as np

import nalp.utils.logging as l
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class OnehotEncoder(Encoder):
    """An OnehotEncoder class is responsible for encoding text into one-hot encodings.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> OnehotEncoder.')

        # Overrides its parent class with any custom arguments if needed
        super(OnehotEncoder, self).__init__()

        # Creates an empty decoder property
        self.decoder = None

        # Creates an empty vocabulary size property
        self.vocab_size = None

        logger.info('Class overrided.')

    @property
    def decoder(self):
        """dict: A decoder dictionary.

        """

        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    @property
    def vocab_size(self):
        """int: The vocabulary size.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    def learn(self, dictionary, reverse_dictionary, vocab_size):
        """Learns an one-hot encoding.

        Args:
            dictionary (dict): The vocabulary to index mapping.
            reverse_dictionary (dict): The index to vocabulary mapping.
            vocab_size (int): The vocabulary size.

        """

        logger.debug('Learning how to encode ...')

        # Creates the encoder property
        self.encoder = dictionary

        # Creates the decoder property
        self.decoder = reverse_dictionary

        # Creates the vocabulary size property
        self.vocab_size = vocab_size

    def encode(self, tokens):
        """Encodes new tokens based on previous learning.

        Args:
            tokens (list): A list of tokens to be encoded.

        Returns:
            A numpy array of encoded tokens.

        """

        logger.debug('Encoding new tokens ...')

        # Checks if enconder actually exists, if not raises an error
        if not self.encoder:
            # Creates the error
            e = 'You need to call learn() prior to encode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Creating an array to hold the one-hot encoded tokens
        encoded_tokens = np.zeros((len(tokens), self.vocab_size), dtype='float32')

        # Iterates through all tokens
        for i, idx in enumerate(tokens):
            # One-hot encodes the token
            encoded_tokens[i, self.encoder[idx]] = 1

        return encoded_tokens

    def decode(self, encoded_tokens):
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens (np.array): A numpy array containing the encoded tokens.

        Returns:
            A list of decoded tokens.

        """

        logger.debug('Decoding encoded tokens ...')

        # Checks if decoder actually exists, if not raises an error
        if not self.decoder:
            # Creates the error
            e = 'You need to call learn() prior to decode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Decoding the tokens
        decoded_tokens = [self.decoder[np.where(encoded_token == 1)[
            0][0]] for encoded_token in encoded_tokens]

        return decoded_tokens
