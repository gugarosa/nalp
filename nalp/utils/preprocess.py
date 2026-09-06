# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Text pre-processing utilities."""

import re
from collections.abc import Callable


def lower_case(s: str) -> str:
    """Return a lowercase copy of the input text.

    Args:
        s: Text to convert.

    Returns:
        Lowercased text.

    """

    return s.lower()


def valid_char(s: str) -> str:
    """Keep ASCII letters, ASCII digits, and whitespace characters.

    Args:
        s: Text to filter.

    Returns:
        Text with characters outside the allowed set removed.

    """

    return re.sub(r"[^A-Za-z0-9\s]", "", s)


def tokenize_to_char(s: str) -> list[str]:
    """Split text into individual characters without normalization.

    Args:
        s: Text to split.

    Returns:
        Characters in their original order, including whitespace.

    """

    return list(s)


def tokenize_to_word(s: str) -> list[str]:
    """Split text into nonempty whitespace-delimited tokens.

    Args:
        s: Text to split.

    Returns:
        Tokens with separating whitespace removed.

    """

    return s.split()


def pipeline(*functions: Callable) -> Callable:
    """Compose preprocessing functions in declaration order.

    Args:
        *functions: Unary callables applied from left to right.

    Returns:
        A callable returning the final result of passing its input through the functions.

    """

    def process(value):
        for function in functions:
            value = function(value)
        return value

    return process


def tokenize(text: str, corpus_type: str) -> list[str]:
    """Lowercase, filter, and tokenize text as characters or words.

    Args:
        text: Source text to normalize and tokenize.
        corpus_type: Tokenization mode, either char or word.

    Returns:
        Tokens from the normalized text.

    Raises:
        RuntimeError: The requested tokenization mode is unsupported.

    """

    tokenizers = {"char": tokenize_to_char, "word": tokenize_to_word}

    try:
        tokenizer = tokenizers[corpus_type]
    except KeyError as error:
        raise RuntimeError(f"`corpus_type` must be `char` or `word`, but got {corpus_type!r}.") from error

    return pipeline(lower_case, valid_char, tokenizer)(text)
