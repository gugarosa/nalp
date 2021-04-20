"""Integer-based encoding.
"""

import numpy as np

import nalp.utils.constants as c
import nalp.utils.logging as l
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class IntegerEncoder(Encoder):
    """An IntegerEncoder class is responsible for encoding text into integers.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> IntegerEncoder.')

        # Overrides its parent class with any custom arguments if needed
        super(IntegerEncoder, self)

        # Creates an empty decoder property
        self.decoder = None

        logger.info('Class overrided.')

    @property
    def decoder(self):
        """dict: A decoder dictionary.

        """

        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder

    def learn(self, dictionary, reverse_dictionary):
        """Learns an integer vectorization encoding.

        Args:
            dictionary (dict): The vocabulary to index mapping.
            reverse_dictionary (dict): The index to vocabulary mapping.

        """

        # Creates the encoder property
        self.encoder = dictionary

        # Creates the decoder property
        self.decoder = reverse_dictionary

    def encode(self, tokens):
        """Encodes new tokens based on previous learning.

        Args:
            tokens (list): A list of tokens to be encoded.

        Returns:
            A numpy array of encoded tokens.

        """

        # Checks if enconder actually exists, if not raises an error
        if not self.encoder:
            # Creates the error
            e = 'You need to call learn() prior to encode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Instantiates a list of empty encoded tokens
        encoded_tokens = []

        # Iterates through every token
        for token in tokens:
            # Checks if token is a list
            if isinstance(token, (np.ndarray, list)):
                # If yes, appends the encoded list
                encoded_tokens.append([self.encoder[t] if t in self.encoder else self.encoder[c.UNK] for t in token])

            # If token is not a list
            else:
                # Checks if token really exists in the vocabulary
                if token in self.encoder:
                    # Concatenates the encoded token
                    encoded_tokens += [self.encoder[token]]

                # If token does not exist in vocabulary
                else:
                    # Concatenates the unknown token
                    encoded_tokens += [self.encoder[c.UNK]]

        # Applies the encoding to the new tokens
        encoded_tokens = np.array(encoded_tokens, dtype=np.int32)

        return encoded_tokens

    def decode(self, encoded_tokens):
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens (np.array): A numpy array containing the encoded tokens.

        Returns:
            A list of decoded tokens.

        """

        # Checks if decoder actually exists, if not raises an error
        if not self.decoder:
            # Creates the error
            e = 'You need to call learn() prior to decode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Instantiates a list of decoded tokens
        decoded_tokens = []

        # Iterates through every token
        for token in encoded_tokens:
            # Checks if token is a list
            if isinstance(token, (np.ndarray, list)):
                # If yes, appends the encoded list
                decoded_tokens.append([self.decoder[t] for t in token])

            # If token is not a list
            else:
                # Concatenates the encoded token
                decoded_tokens += [self.decoder[token]]

        return decoded_tokens
