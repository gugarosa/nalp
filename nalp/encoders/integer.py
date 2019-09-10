import numpy as np

import nalp.utils.logging as l
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class IntegerEncoder(Encoder):
    """An Integer class, responsible for encoding text into integers.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> IntegerEncoder.')

        # Overrides its parent class with any custom arguments if needed
        super(IntegerEncoder, self).__init__()

        logger.info('Class overrided.')

    def learn(self, dictionary):
        """Learns an integer vectorization encoding.

        Args:
            dictionary (dict): The vocabulary to index mapping.

        """

        logger.debug('Learning how to encode ...')

        self.encoder = dictionary

    def encode(self, tokens):
        """Encodes new tokens based on previous learning.

        Args:
            tokens (list): A list of tokens to be encoded.
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
        encoded_tokens = np.array([self.encoder[c] for c in tokens])

        return encoded_tokens
