import feelyng.utils.logging as l
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = l.get_logger(__name__)


def learn_tfidf(sentences, max_features=100):
    """

    Args:
        sentences (df): A Panda's dataframe column holding sentences to be fitted.
        max_features (int): Maximum number of features to be fitted.

    Returns:
        A TFIDF Vectorizer object.

    """

    # Creates a TFIDF vectorizer
    tfidf = TfidfVectorizer(max_features=max_features)

    # Fits sentences on it
    logger.info('Fitting TFIDF ...')
    tfidf.fit(sentences)
    logger.info('TFIDF fitted.')

    return tfidf


def encode_tfidf(tfidf, sentences):
    """

    Args:
        tfidf (TfidfVectorizer): A TFIDF Vectorizer object.
        sentences (df): A Panda's dataframe column holding sentences to be encoded.

    Returns:
        An encoded TFIDF numpy array.

    """

    logger.debug('TFIDF encoding size: (' + str(sentences.size) + ', ' + str(tfidf.idf_.shape[0]) + ')')

    # Transform sentences into TFIDF encoding (only if it has been previously fitted)
    logger.info('Encoding data ...')
    X = tfidf.transform(sentences)
    
    # Creates a numpy array to hold encoded TFIDF
    encoded_X = np.zeros((sentences.size, tfidf.idf_.shape[0]))

    # Apply encoded TFIDF to numpy array
    X.toarray(out=encoded_X)
    logger.info('Encoding finished.')

    return encoded_X    