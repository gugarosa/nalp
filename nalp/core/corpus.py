# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Corpus-related class."""

from collections import Counter

import nalp.utils.constants as c


class Corpus:
    """Hold tokenized data and deterministic vocabulary mappings."""

    def __init__(self, min_frequency: int = 1) -> None:
        """Initialize the containers populated by concrete corpus classes.

        Args:
            min_frequency: Minimum corpus-wide count required to retain a token during frequency filtering.

        """

        self.min_frequency = min_frequency

        self.tokens: list[str] = []
        self.vocab: list[str] = []
        self.vocab_size = 0
        self.vocab_index: dict[str, int] = {}
        self.index_vocab: dict[int, str] = {}

    def _check_token_frequency(self) -> None:
        tokens_frequency = Counter(self.tokens)

        # Keep caller-owned token lists connected to the corpus
        self.tokens[:] = [token if tokens_frequency[token] >= self.min_frequency else c.UNK for token in self.tokens]

    def _build(self) -> None:
        self.vocab = sorted(set(self.tokens).union({c.UNK}))
        self.vocab_size = len(self.vocab)

        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
