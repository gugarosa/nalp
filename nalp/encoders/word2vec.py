import multiprocessing

import numpy as np
from gensim.models.word2vec import Word2Vec as W2V

import nalp.utils.logging as l
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class Word2vecEncoder(Encoder):
    """A Word2vecEncoder class is responsible for learning a Word2Vec encode and
    further encoding new data.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> Word2vecEncoder.')

        # Overrides its parent class with any custom arguments if needed
        super(Word2vecEncoder, self).__init__()

        logger.info('Class overrided.')

    def learn(self, tokens, max_features=128, window_size=5, min_count=1, algorithm=0, learning_rate=0.01, iterations=10):
        """Learns a Word2Vec representation based on the its methodology.

        One can use CBOW or Skip-gram algorithm for the learning procedure.

        Args:
            tokens (list): A list of tokens.
            max_features (int): Maximum number of features to be fitted.
            window_size (int): Maximum distance between current and predicted word.
            min_count (int): Minimum count of words for its use.
            algorithm (bool): 1 for skip-gram, while 0 for CBOW.
            learning_rate (float): Starting value of the learning procedure learning rate.
            iterations (int): Number of iterations.

        """

        logger.debug('Learning how to encode ...')

        # Creates a Word2Vec model
        self.encoder = W2V(sentences=tokens, size=max_features, window=window_size, min_count=min_count,
                            sg=algorithm, alpha=learning_rate, iter=iterations, workers=multiprocessing.cpu_count())

    def encode(self, tokens, max_tokens=10):
        """Encodes the data into a Word2Vec representation.

        Args:
            tokens (list): A list of tokens to be encoded.
            max_tokens (int): Maximum amount of tokens per sentence.

        """

        logger.debug('Encoding new tokens ...')

        # Checks if enconder actually exists, if not raises an error
        if not self.encoder:
            # Creates the error
            e = 'You need to call learn() prior to encode() method.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # Gets the actual word vectors from Word2Vec class
        wv = self.encoder.wv

        # Creates an encoded tokens variable to hold encoded data
        encoded_tokens = np.zeros((len(tokens), max_tokens, self.encoder.vector_size))

        # Iterate through all sentences
        for i in range(0, len(tokens)):
            # For each sentence, iterate over its tokens
            for t, token in enumerate(tokens[i]):
                # If token index exceed maximum length, break the loop
                if t >= max_tokens:
                    break
                # Else, store its word vector value to a new variable
                encoded_tokens[i, t, :] = wv[token]

        return encoded_tokens
