import multiprocessing

import feelyng.utils.logging as l
import numpy as np
from gensim.models.word2vec import Word2Vec

logger = l.get_logger(__name__)


def learn_word2vec(sentences, max_features=128, window_size=5, min_count=1, algorithm=0, learning_rate=0.01, iterations=10):
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

    Returns:
        A Word2Vec object.

    """

    # Creates a Word2Vec model
    logger.info('Fitting Word2Vec ...')
    word2vec = Word2Vec(sentences=sentences, size=max_features, window=window_size, min_count=min_count,
                        sg=algorithm, alpha=learning_rate, iter=iterations, workers=multiprocessing.cpu_count())
    logger.info('Word2Vec fitted.')

    return word2vec


def encode_word2vec(word2vec, sentences, max_tokens=10):
    """Actually encodes the data into a Word2Vec representation.

    Args:
        word2vec (Word2Vec): A Word2Vec object.
        sentences (df): A Panda's dataframe column holding sentences to be encoded.
        max_tokens (int): Maximum amount of tokens per sentence.

    Returns:
        An encoded Word2Vec numpy array.

    """

    logger.debug('Word2vec encoding size: (' + str(sentences.size) +
                 ', ' + str(max_tokens) + ', ' + str(word2vec.vector_size) + ')')

    # Get actual word vectors from Word2Vec class
    wv = word2vec.wv

    # Creates an encoded_X variable to hold encoded data
    logger.info('Encoding data ...')
    encoded_X = np.zeros((sentences.size, max_tokens, word2vec.vector_size))

    # Iterate through all sentences
    for i in range(0, sentences.size):
        # For each sentence, iterate over its tokens
        for t, token in enumerate(sentences[i]):
            # If token index exceed maximum length, break the loop
            if t >= max_tokens:
                break
            # Else, store its word vector value to a new variable
            encoded_X[i, t, :] = wv[token]

    logger.info('Encoding finished.')

    return encoded_X
