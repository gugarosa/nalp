import nalp.utils.logging as l
import numpy as np
from nalp.core.encoder import Encoder
from sklearn.feature_extraction.text import CountVectorizer

logger = l.get_logger(__name__)


class Count(Encoder):

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> Count.')

        # Overrides its parent class with any custom arguments if needed
        super(Count, self).__init__()

        logger.info('Class overrided.')

    def learn(self, sentences, max_features=100):
        """Learns a CountVectorizer representation based on the words' counting.

        Args:
            sentences (df): A Panda's dataframe column holding sentences to be fitted.
            max_features (int): Maximum number of features to be fitted.

        """

        logger.debug('Running public method: learn().')

        # Creates a CountVectorizer object
        self._encoder = CountVectorizer(max_features=max_features,
                                       preprocessor=lambda p: p, tokenizer=lambda t: t)

        # Fits sentences onto it
        self._encoder.fit(sentences)

    def encode(self, sentences):
        """Actually encodes the data into a CountVectorizer representation.

        Args:
            sentences (df): A Panda's dataframe column holding sentences to be encoded.

        """

        logger.debug('Running public method: encode().')

        # Checks if enconder actually exists, if not raises a RuntimeError
        if not self._encoder:
            e = 'You need to call learn() prior to encode() method.'
            logger.error(e)
            raise RuntimeError(e)

        # Logging some important information
        logger.debug(f'Size: ({sentences.size}, {self._encoder.max_features}).')

        # Transforms sentences into CountVectorizer encoding (only if it has been previously fitted)
        X = self._encoder.transform(sentences)

        # Applies encoded CountVectorizer to a numpy array
        self._encoded_data = X.toarray()
