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

        if not self.encoder:
            e = 'You need to call learn() prior to encode() method.'

            logger.error(e)

            raise RuntimeError(e)

        encoded_tokens = []

        for token in tokens:
            if isinstance(token, (np.ndarray, list)):
                encoded_tokens.append([self.encoder[t] if t in self.encoder else self.encoder[c.UNK] for t in token])

            else:
                if token in self.encoder:
                    encoded_tokens += [self.encoder[token]]

                else:
                    encoded_tokens += [self.encoder[c.UNK]]

        encoded_tokens = np.array(encoded_tokens, dtype=np.int32)

        return encoded_tokens

    def decode(self, encoded_tokens):
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens (np.array): A numpy array containing the encoded tokens.

        Returns:
            A list of decoded tokens.

        """

        if not self.decoder:
            e = 'You need to call learn() prior to decode() method.'

            logger.error(e)

            raise RuntimeError(e)

        decoded_tokens = []

        for token in encoded_tokens:
            if isinstance(token, (np.ndarray, list)):
                decoded_tokens.append([self.decoder[t] for t in token])

            else:
                decoded_tokens += [self.decoder[token]]

        return decoded_tokens
