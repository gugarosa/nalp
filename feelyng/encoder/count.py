import feelyng.utils.logging as l
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = l.get_logger(__name__)


def learn_count(sentences, max_features=100):
    """

    Args:

    Returns:

    """

    # Creates a Count vectorizer
    count = CountVectorizer(max_features=max_features)
    logger.debug('Count created')

    # Fits sentences on it
    count.fit(sentences)
    logger.debug('Count fitted')

    return count


def encode_count(count, sentences):
    """

    Args:

    Returns:

    """

    logger.debug('Count encoding size: (' + str(sentences.size) + ', ' + str(count.max_features) + ')')
    # Creates an encoded variable to hold encoded text
    encoded_X = np.zeros((sentences.size, count.max_features))

    # Iterate through all sentences 
    for i in range(1, sentences.size):
        # Encode each sentence into an array
        encoded_X[i] = (count.transform([sentences[i]])).toarray()
    


    logger.debug('Count encoding finished')

    return encoded_X    