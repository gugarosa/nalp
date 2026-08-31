"""Sentence-related corpus."""

from collections import Counter
from itertools import chain
from pathlib import Path

import nalp.utils.constants as c
import nalp.utils.preprocess as p
from nalp.core import Corpus
from nalp.utils import loader


class SentenceCorpus(Corpus):
    """A SentenceCorpus class is used to defined the first step of the workflow.

    It serves to load the raw sentences, pre-process them and create their tokens and
    vocabulary.

    """

    def __init__(
        self,
        tokens: list[list[str]] | None = None,
        from_file: str | Path | None = None,
        corpus_type: str = "char",
        min_frequency: int = 1,
        max_pad_length: int | None = None,
        sos_eos_tokens: bool = True,
    ) -> None:
        """Initialization method.

        Args:
            tokens: A list of tokens.
            from_file: An input file to load the sentences.
            corpus_type: The desired type to tokenize the sentences. Should be `char` or `word`.
            min_frequency: Minimum frequency of individual tokens.
            max_pad_length: Maximum length to pad the tokens.
            sos_eos_tokens: Whether start-of-sentence and end-of-sentence tokens should be used.

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
        """Cuts tokens that do not meet a minimum frequency value."""

        tokens_frequency = Counter(chain.from_iterable(self.tokens))

        for sentence in self.tokens:
            sentence[:] = [
                token if tokens_frequency[token] >= self.min_frequency else c.UNK
                for token in sentence
            ]

    def _pad_token(self, max_pad_length: int | None, sos_eos_tokens: bool) -> None:
        """Pads the tokens into a fixed length.

        Args:
            max_pad_length: Maximum length to pad the tokens.
            sos_eos_tokens: Whether start-of-sentence and end-of-sentence tokens should be used.

        """

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
        """Builds the vocabulary based on the tokens."""

        self.vocab = sorted(set(chain.from_iterable(self.tokens)).union({c.UNK}))
        self.vocab_size = len(self.vocab)

        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
