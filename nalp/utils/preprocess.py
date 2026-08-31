"""Text pre-processing utilities."""

import re
from collections.abc import Callable


def lower_case(s: str) -> str:
    """Return a lower-case copy of ``s``."""

    return s.lower()


def valid_char(s: str) -> str:
    """Remove characters outside letters, digits, and whitespace."""

    return re.sub(r"[^A-Za-z0-9\s]", "", s)


def tokenize_to_char(s: str) -> list[str]:
    """Split text into characters."""

    return list(s)


def tokenize_to_word(s: str) -> list[str]:
    """Split text on whitespace."""

    return s.split()


def pipeline(*functions: Callable) -> Callable:
    """Compose preprocessing functions in declaration order."""

    def process(value):
        for function in functions:
            value = function(value)
        return value

    return process


def tokenize(text: str, corpus_type: str) -> list[str]:
    """Lowercase, filter, and tokenize text as characters or words."""

    tokenizers = {"char": tokenize_to_char, "word": tokenize_to_word}
    try:
        tokenizer = tokenizers[corpus_type]
    except KeyError as error:
        raise RuntimeError("Corpus type should be `char` or `word`.") from error

    return pipeline(lower_case, valid_char, tokenizer)(text)
