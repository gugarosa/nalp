import multiprocessing

import nalp.utils.logging as l
import numpy as np
from gensim.models.word2vec import Word2Vec as W2V
from nalp.core.encoder import Encoder

logger = l.get_logger(__name__)


class Word2Vec(Encoder):
    """A Word2Vec class, responsible for learning a Word2Vec encode and
    further encoding new data.

    """

    def __init__(self):
        """Initizaliation method.

        """

        logger.info('Overriding class: Encoder -> Word2Vec.')

        # Overrides its parent class with any custom arguments if needed
        super(Word2Vec, self).__init__()

        logger.info('Class overrided.')

    def learn(self, sentences, max_features=128, window_size=5, min_count=1, algorithm=0, learning_rate=0.01, iterations=10):
        """Learns a Word2Vec representation based on the its methodology.
        One can use CBOW or Skip-gram algorithm for the learning procedure.

        Args:
            sentences (df): A Panda's dataframe column holding sentences to be fitted.
            max_features (int): Maximum number of features to be fitted.
            window_size (int): Maximum distance between current and predicted word.
            min_count (int): Minimum count of words for its use.
            algorithm (bool): 1 for skip-gram, while 0 for CBOW.
            learning_rate (float): Starting value of the learning procedure learning rate.
            iterations (int): Number of iterations.

        """

        logger.debug('Running public method: learn().')

        # Creates a Word2Vec model
        self.encoder = W2V(sentences=sentences, size=max_features, window=window_size, min_count=min_count,
                            sg=algorithm, alpha=learning_rate, iter=iterations, workers=multiprocessing.cpu_count())

    def encode(self, sentences, max_tokens=10):
        """Actually encodes the data into a Word2Vec representation.

        Args:
            sentences (df): A Panda's dataframe column holding sentences to be encoded.
            max_tokens (int): Maximum amount of tokens per sentence.

        """

        logger.debug('Running public method: encode().')

        # Checks if enconder actually exists, if not raises a RuntimeError
        if not self.encoder:
            e = 'You need to call learn() prior to encode() method.'
            logger.error(e)
            raise RuntimeError(e)

        # Logging some important information
        logger.debug(
            f'Size: ({sentences.size}, {max_tokens}, {self.encoder.vector_size}).')

        # Get actual word vectors from Word2Vec class
        wv = self.encoder.wv

        # Creates an encoded_X variable to hold encoded data
        self.encoded_data = np.zeros(
            (sentences.size, max_tokens, self.encoder.vector_size))

        # Iterate through all sentences
        for i in range(0, sentences.size):
            # For each sentence, iterate over its tokens
            for t, token in enumerate(sentences[i]):
                # If token index exceed maximum length, break the loop
                if t >= max_tokens:
                    break
                # Else, store its word vector value to a new variable
                self.encoded_data[i, t, :] = wv[token]
