# Copyright (c) 2019-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Text-related corpus."""

from pathlib import Path

import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus
from nalp.utils import loader


class TextCorpus(Corpus):
    """Build token and vocabulary mappings for contiguous text."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        from_file: str | Path | None = None,
        corpus_type: str = "char",
        min_frequency: int = 1,
    ) -> None:
        """Build a corpus from supplied tokens or a UTF-8 text file.

        Reuse a nonempty supplied token list and replace infrequent tokens in place.
        File input is lowercased and filtered to ASCII letters, digits, and whitespace.
        Character tokenization preserves raw line endings.

        Args:
            tokens: Pre-tokenized data reused when nonempty.
            from_file: Source path used when tokens is None or empty.
            corpus_type: File tokenization mode, either char or word.
            min_frequency: Minimum token count before replacement with the unknown-token marker.

        Raises:
            OSError: The source file cannot be read.
            UnicodeDecodeError: The source file is not valid UTF-8.
            RuntimeError: The selected file tokenization mode is unsupported.

        """

        super().__init__(min_frequency=min_frequency)

        if not tokens:
            text = loader.load_txt(from_file)
            self.tokens = p.tokenize(text, corpus_type)
        else:
            self.tokens = tokens

        self._check_token_frequency()
        self._build()
