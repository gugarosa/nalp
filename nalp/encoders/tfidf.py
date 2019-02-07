import nalp.utils.logging as l
import numpy as np
from nalp.core.encoder import Encoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = l.get_logger(__name__)


class TFIDF(Encoder):
    """A TFIDF class, responsible for learning a TfidfVectorizer encode and
    further encoding new data.

    Methods:
        learn(sentences, max_features): Learns a TfidfVectorizer representation.
        encode(sentences): Encodes the data into a TfidfVectorizer representation.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> TFIDF.')

        # Overrides its parent class with any custom arguments if needed
        super(TFIDF, self).__init__()

        logger.info('Class overrided.')

    def learn(self, sentences, max_features=100):
        """Learns a TFIDF representation based on the words' frequency.

        Args:
            sentences (df): A Panda's dataframe column holding sentences to be fitted.
            max_features (int): Maximum number of features to be fitted.

        """

        logger.debug('Running public method: learn().')

        # Creates a TfidfVectorizer object
        self._encoder = TfidfVectorizer(max_features=max_features,
                                        preprocessor=lambda p: p, tokenizer=lambda t: t)

        # Fits sentences onto it
        self._encoder.fit(sentences)

    def encode(self, sentences):
        """Actually encodes the data into a TfidfVectorizer representation.

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
        logger.debug(
            f'Size: ({sentences.size}, {self._encoder.max_features}).')

        # Transforms sentences into TfidfVectorizer encoding (only if it has been previously fitted)
        X = self._encoder.transform(sentences)

        # Applies encoded TfidfVectorizer to a numpy array
        self._encoded_data = X.toarray()
