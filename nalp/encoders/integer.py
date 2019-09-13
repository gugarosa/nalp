import numpy as np

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
        super(IntegerEncoder, self).__init__()

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

        logger.debug('Learning how to encode ...')

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

        logger.debug('Encoding new tokens ...')

        # Checks if enconder actually exists, if not raises an error
        if not self.encoder:
            # Creates the error
            e = 'You need to call learn() prior to encode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Applies the encoding to the new tokens
        encoded_tokens = np.array([self.encoder[t] for t in tokens])

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
        decoded_tokens = [self.decoder[t] for t in encoded_tokens]

        return decoded_tokens
