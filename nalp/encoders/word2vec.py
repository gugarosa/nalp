"""Word2vec encoding.
"""

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

        super(Word2vecEncoder, self)

        logger.info('Class overrided.')

    def learn(self, tokens, max_features=128, window_size=5, min_count=1,
              algorithm=0, learning_rate=0.01, iterations=1000):
        """Learns a Word2Vec representation based on the its methodology.

        One can use CBOW or Skip-gram algorithm for the learning procedure.

        Args:
            tokens (list): A list of tokens.
            max_features (int): Maximum number of features to be fitted.
            window_size (int): Maximum distance between current and predicted word.
            min_count (int): Minimum count of words for its use.
            algorithm (bool): 1 for skip-gram, while 0 for CBOW.
            learning_rate (float): Value of the learning rate.
            iterations (int): Number of iterations.

        """

        self.encoder = W2V(sentences=[tokens], vector_size=max_features, window=window_size, min_count=min_count,
                           sg=algorithm, alpha=learning_rate, epochs=iterations,
                           workers=multiprocessing.cpu_count())

    def encode(self, tokens):
        """Encodes the data into a Word2Vec representation.

        Args:
            tokens (list): A list of tokens to be encoded.

        """

        if not self.encoder:
            e = 'You need to call learn() prior to encode() method.'

            logger.error(e)

            raise RuntimeError(e)

        # Gets the actual word vectors from Word2Vec class
        wv = self.encoder.wv

        # Creates an encoded tokens variable to hold encoded data
        encoded_tokens = np.zeros((len(tokens), self.encoder.vector_size))

        for i, token in enumerate(tokens):
            encoded_tokens[i, :] = wv[token]

        return encoded_tokens

    def decode(self, encoded_tokens):
        """Decodes the encoding back to tokens.

        Args:
            encoded_tokens (np.array): A numpy array containing the encoded tokens.

        Returns:
            A list of decoded tokens.

        """

        if not self.encoder:
            e = 'You need to call learn() prior to decode() method.'

            logger.error(e)

            raise RuntimeError(e)

        decoded_tokens = [self.encoder.wv.most_similar(positive=[t])[0][0] for t in encoded_tokens]

        return decoded_tokens
