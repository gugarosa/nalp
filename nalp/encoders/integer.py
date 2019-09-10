import nalp.utils.logging as l
import numpy as np
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

    def learn(self, corpus):
        """
        """

        logger.debug('Learning how to encode ...')

        self.encoder = corpus.vocab_index

    def encode(self, tokens):
        """
        """

        logger.debug('Encoding new tokens ...')

        #
        encoded_tokens = np.array([self.encoder[c] for c in tokens])

        return encoded_tokens