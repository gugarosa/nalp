"""Pre-processing functions.
"""

import re

import nltk

import nalp.utils.logging as l

logger = l.get_logger(__name__)


def lower_case(s):
    """Transforms an input string into its lower case version.

    Args:
        s (str): Input string.

    Returns:
        Lower case of 's'.

    """

    return s.lower()


def valid_char(s):
    """Validates the input string characters.

    Args:
        s (str): Input string.

    Returns:
        String 's' after validation.

    """

    return re.sub('[^a-zA-z0-9\s]', '', s)


def tokenize_to_char(s):
    """Tokenizes a text to characters array.

    Args:
        s (str): Input string.

    Returns:
        Array of tokenized characters.

    """

    return list(s)


def tokenize_to_word(s):
    """Tokenizes a text to words array.

    Args:
        s (str): Input string.

    Returns:
        Array of tokenized words.

    """

    return nltk.regexp_tokenize(s, pattern='\s+', gaps=True)


def pipeline(*func):
    """Creates a pre-processing pipeline.

    Args:
        *func (callable): Functions pointers.

    Returns:
        A created pre-processing pipeline for further use.

    """

    def process(x):
        for f in func:
            x = f(x)

        return x

    logger.debug('Pipeline created with %s.', str(func))

    return process
