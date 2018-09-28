import feelyng.utils.logging as l
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = l.get_logger(__name__)


def learn_count(sentences, max_features=100):
    """

    Args:
        sentences (df): A Panda's dataframe column holding sentences to be fitted.
        max_features (int): Maximum number of features to be fitted.

    Returns:
        A CountVectorizer object.

    """

    # Creates a Count vectorizer
    count = CountVectorizer(max_features=max_features,
                            preprocessor=lambda p: p, tokenizer=lambda t: t)

    # Fits sentences on it
    logger.info('Learning CountVectorizer ...')
    count.fit(sentences)
    logger.info('CountVectorizer learned.')

    return count


def encode_count(count, sentences):
    """

    Args:
        count (CountVectorizer): A CountVectorizer object.
        sentences (df): A Panda's dataframe column holding sentences to be encoded.

    Returns:
        An encoded CountVectorizer numpy array.

    """

    logger.info('CountVectorizer encoding size: (' +
                 str(sentences.size) + ', ' + str(count.max_features) + ')')

    # Transform sentences into CountVectorizer encoding (only if it has been previously fitted)
    logger.info('Encoding data ...')
    X = count.transform(sentences)

    # Apply encoded TFIDF to a numpy array
    encoded_X = X.toarray()
    logger.info('Encoding finished.')

    return encoded_X
