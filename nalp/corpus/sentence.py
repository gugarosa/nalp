# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Sentence-related corpus."""

from collections import Counter
from itertools import chain
from pathlib import Path

import nalp.utils.constants as c
import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus
from nalp.utils import loader


class SentenceCorpus(Corpus):
    """Tokenize, filter, and pad separate sentences into a shared vocabulary."""

    def __init__(
        self,
        tokens: list[list[str]] | None = None,
        from_file: str | Path | None = None,
        corpus_type: str = "char",
        min_frequency: int = 1,
        max_pad_length: int | None = None,
        sos_eos_tokens: bool = True,
    ) -> None:
        """Build a padded sentence corpus from tokens or a UTF-8 text file.

        Reuse a nonempty supplied outer list and change its contents during filtering, padding, and truncation.
        File input is tokenized per line, and optional boundary markers are added after padding.

        Args:
            tokens: Pre-tokenized sentences reused when the outer list is nonempty.
            from_file: Source path used when tokens is None or empty.
            corpus_type: File tokenization mode, either char or word.
            min_frequency: Minimum corpus-wide token count before replacement with the unknown-token marker.
            max_pad_length: Sentence width before boundaries, with None or 0 selecting the longest sentence.
            sos_eos_tokens: Whether to prepend a start marker and append an end marker after padding.

        Raises:
            OSError: The source file cannot be read.
            UnicodeDecodeError: The source file is not valid UTF-8.
            RuntimeError: The selected file tokenization mode is unsupported.

        """

        super().__init__(min_frequency=min_frequency)

        if not tokens:
            sentences = loader.load_txt(from_file).splitlines()
            self.tokens = [p.tokenize(sentence, corpus_type) for sentence in sentences]
        else:
            self.tokens = tokens

        self._check_token_frequency()
        self._pad_token(max_pad_length, sos_eos_tokens)
        self._build()

    def _check_token_frequency(self) -> None:
        tokens_frequency = Counter(chain.from_iterable(self.tokens))

        for sentence in self.tokens:
            sentence[:] = [token if tokens_frequency[token] >= self.min_frequency else c.UNK for token in sentence]

    def _pad_token(self, max_pad_length: int | None, sos_eos_tokens: bool) -> None:
        if not max_pad_length:
            max_pad_length = len(max(self.tokens, key=lambda t: len(t)))

        for i, _ in enumerate(self.tokens):
            length_diff = max_pad_length - len(self.tokens[i])

            if length_diff > 0:
                self.tokens[i] += [c.PAD] * length_diff
            else:
                self.tokens[i] = self.tokens[i][:max_pad_length]

            if sos_eos_tokens:
                self.tokens[i].insert(0, c.SOS)
                self.tokens[i].append(c.EOS)

    def _build(self) -> None:
        self.vocab = sorted(set(chain.from_iterable(self.tokens)).union({c.UNK}))
        self.vocab_size = len(self.vocab)

        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
