import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nalp.utils.logging as l
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class TfidfEncoder(Encoder):
    """A TfidfEncoder class is responsible for learning a TfidfVectorizer encoding and
    further encoding new data.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> TfidfEncoder.')

        # Overrides its parent class with any custom arguments if needed
        super(TfidfEncoder, self).__init__()

        logger.info('Class overrided.')

    def learn(self, tokens, top_tokens=100):
        """Learns a TfidfVectorizer representation based on the tokens' frequency.

        Args:
            tokens (list): A list of tokens.
            top_tokens (int): Maximum number of top tokens to be learned.

        """

        logger.debug('Learning how to encode ...')

        # Creates a TfidfVectorizer object
        self.encoder = TfidfVectorizer(max_features=top_tokens)

        # Fits the tokens
        self.encoder.fit(tokens)

    def encode(self, tokens):
        """Encodes the data into a TfidfVectorizer representation.

        Args:
            tokens (list): A list of tokens to be encoded.

        Returns:
            A numpy array containing the encoded tokens.

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
        encoded_tokens = (self.encoder.transform(tokens)).toarray()

        return encoded_tokens

    def decode(self, encoded_tokens):
        """Decodes the TfidfVectorizer representation back to tokens.

        Args:
            encoded_tokens (np.array): A numpy array containing the encoded tokens.

        Returns:
            A list of decoded tokens.

        """

        logger.debug('Decoding encoded tokens ...')

        # Checks if enconder actually exists, if not raises an error
        if not self.encoder:
            # Creates the error
            e = 'You need to call learn() prior to decode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Decoding the tokens
        decoded_tokens = self.encoder.inverse_transform(encoded_tokens)

        # Joining every list of decoded tokens into a sentence
        decoded_tokens = [' '.join(list(d)) for d in decoded_tokens]

        return decoded_tokens
