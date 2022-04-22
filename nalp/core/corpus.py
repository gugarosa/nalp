"""Corpus-related class.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import nalp.utils.constants as c
import nalp.utils.preprocess as p
from nalp.utils import logging

logger = logging.get_logger(__name__)


class Corpus:
    """A Corpus class is used to defined the first step of the workflow.

    It serves as a basis class to load raw text, audio and sentences.

    Note that this class only provides basic properties and methods that are invoked
    by its childs, thus, it should not be instantiated.

    """

    def __init__(self, min_frequency: Optional[int] = 1) -> None:
        """Initialization method."""

        # Minimum token frequency
        self.min_frequency = min_frequency

    @property
    def tokens(self) -> List[str]:
        """List of input tokens."""

        return self._tokens

    @tokens.setter
    def tokens(self, tokens: List[str]) -> None:
        self._tokens = tokens

    @property
    def vocab(self) -> List[str]:
        """Vocabulary tokens."""

        return self._vocab

    @vocab.setter
    def vocab(self, vocab: List[str]) -> None:
        self._vocab = vocab

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_index(self) -> Dict[str, Any]:
        """Maps vocabulary tokens to indexes."""

        return self._vocab_index

    @vocab_index.setter
    def vocab_index(self, vocab_index: Dict[str, Any]) -> None:
        self._vocab_index = vocab_index

    @property
    def index_vocab(self) -> Dict[str, Any]:
        """Maps indexes to vocabulary tokens."""

        return self._index_vocab

    @index_vocab.setter
    def index_vocab(self, index_vocab: Dict[str, Any]) -> None:
        self._index_vocab = index_vocab

    @property
    def min_frequency(self) -> int:
        """Minimum token frequency."""

        return self._min_frequency

    @min_frequency.setter
    def min_frequency(self, min_frequency: int) -> None:
        self._min_frequency = min_frequency

    def _create_tokenizer(self, corpus_type: str) -> callable:
        """Creates a tokenizer based on the input type.

        Args:
            corpus_type: A type to create the tokenizer. Should be `char` or `word`.

        Returns:
            (callable): The created tokenizer.

        """

        if corpus_type not in ["char", "word"]:
            e = "Corpus type should be `char` or `word`."

            logger.error(e)

            raise RuntimeError(e)

        if corpus_type == "char":
            return p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_char)

        return p.pipeline(p.lower_case, p.valid_char, p.tokenize_to_word)

    def _check_token_frequency(self) -> None:
        """Cuts tokens that do not meet a minimum frequency value."""

        # Calculates the frequency of tokens
        tokens_frequency = Counter(self.tokens)

        # Iterates over every possible sentence
        # Using index is a caveat due to lists immutable property
        for i, _ in enumerate(self.tokens):
            # If frequency of token is smaller than minimum frequency
            if tokens_frequency[self.tokens[i]] < self.min_frequency:
                # Replaces with an unknown token
                self.tokens[i] = c.UNK

    def _build(self) -> None:
        """Builds the vocabulary based on the tokens."""

        # Creates the vocabulary
        self.vocab = sorted(set(self.tokens).union({c.UNK}))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes and vice-versa
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
