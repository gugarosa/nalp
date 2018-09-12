import re

import feelyng.utils.logging as l
import nltk

logger = l.get_logger(__name__)


def lower_case(s):
    """Transforms an input string into its lower case version.

    Args:
        s (str): input string

    Returns:
        Lower case of 's'.

    """

    return s.lower()


def valid_char(s):
    """Validates the input string characters.

    Args:
        s (str): input string

    Returns:
        String 's' after validation.

    """

    return re.sub('[^a-zA-z0-9\s]', '', s)


def sentence_to_word(s):
    """Tokenizes a sentence to words array.

    Args:
        s (str): input string

    Returns:
        Array of tokenized words.

    """

    return nltk.word_tokenize(s)


def pipeline(*func):
    """Creates a pre-processing pipeline.

    Args:
        func (*function): function pointer.

    Returns:
        A created pre-processing pipeline for further use.

    """

    def process(x):
        # Iterate over every argument function
        for f in func:
            # Apply function to input
            x = f(x)
        return x

    logger.info('Pipeline created with ' + str(func) + '.')
    return process
