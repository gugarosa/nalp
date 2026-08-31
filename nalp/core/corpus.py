"""Corpus-related class.
"""

from collections import Counter

import nalp.utils.constants as c


class Corpus:
    """A Corpus class is used to defined the first step of the workflow.

    It serves as a basis class to load raw text, audio and sentences.

    Note that this class only provides basic attributes and methods that are invoked
    by its childs, thus, it should not be instantiated.

    """

    def __init__(self, min_frequency: int = 1) -> None:
        """Initialization method."""

        self.min_frequency = min_frequency
        self.tokens: list[str] = []
        self.vocab: list[str] = []
        self.vocab_size = 0
        self.vocab_index: dict[str, int] = {}
        self.index_vocab: dict[int, str] = {}

    def _check_token_frequency(self) -> None:
        """Cuts tokens that do not meet a minimum frequency value."""

        tokens_frequency = Counter(self.tokens)
        self.tokens[:] = [
            token if tokens_frequency[token] >= self.min_frequency else c.UNK
            for token in self.tokens
        ]

    def _build(self) -> None:
        """Builds the vocabulary based on the tokens."""

        self.vocab = sorted(set(self.tokens).union({c.UNK}))
        self.vocab_size = len(self.vocab)

        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
