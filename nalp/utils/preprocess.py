"""Pre-processing functions.
"""

import re
from typing import List, Union

import nltk

from nalp.utils import logging

logger = logging.get_logger(__name__)


def lower_case(s: str) -> str:
    """Transforms an input string into its lower case version.

    Args:
        s: Input string.

    Returns:
        (str): Lower case of 's'.

    """

    return s.lower()


def valid_char(s: str) -> str:
    """Validates the input string characters.

    Args:
        s: Input string.

    Returns:
        (str): String 's' after validation.

    """

    return re.sub("[^a-zA-z0-9\s]", "", s)


def tokenize_to_char(s: str) -> List[str]:
    """Tokenizes a text to characters array.

    Args:
        s: Input string.

    Returns:
        List[str]: Tokenized characters.

    """

    return list(s)


def tokenize_to_word(s: str) -> List[str]:
    """Tokenizes a text to words array.

    Args:
        s: Input string.

    Returns:
        (List[str]): Tokenized words.

    """

    return nltk.regexp_tokenize(s, pattern="\s+", gaps=True)


def pipeline(*func: callable) -> callable:
    """Creates a pre-processing pipeline.

    Args:
        *func: Functions pointers.

    Returns:
        (callable): Pre-processing pipeline for further use.

    """

    def process(x: str) -> Union[str, List[str]]:
        for f in func:
            x = f(x)

        return x

    logger.debug("Pipeline created with %s.", str(func))

    return process
