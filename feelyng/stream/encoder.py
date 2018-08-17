import feelyng.utils.logging as l
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = l.get_logger(__name__)


def learn_tfidf(sentences, max_features=100):
    """

    Args:

    Returns:

    """

    # Creates a TFIDF vectorizer
    tfidf = TfidfVectorizer(max_features=max_features)
    logger.debug('TFIDF created')

    # Fits sentences on it
    tfidf.fit(sentences)
    logger.debug('TFIDF fitted')

    return tfidf


def encode_tfidf(tfidf, sentences):
    """

    Args:

    Returns:

    """

    logger.debug('TFIDF encoding size: (' + str(sentences.size) + ', ' + str(tfidf.idf_.shape[0]) + ')')
    # Creates an encoded variable to hold encoded text
    encoded_X = np.zeros((sentences.size, tfidf.idf_.shape[0]))

    # Iterate through all sentences
    for i in range(1, sentences.size):
        # Encode each sentence into an array
        encoded_X[i] = (tfidf.transform([sentences[i]])).toarray()

    logger.debug('TFIDF encoding finished')

    return encoded_X

