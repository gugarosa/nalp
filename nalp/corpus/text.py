"""Text-related corpus.
"""

from pathlib import Path

import nalp.utils.preprocess as p
from nalp.core import Corpus


class TextCorpus(Corpus):
    """A TextCorpus class is used to defined the first step of the workflow.

    It serves to load the raw text, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        from_file: str | Path | None = None,
        corpus_type: str = "char",
        min_frequency: int = 1,
    ) -> None:
        """Initialization method.

        Args:
            tokens: A list of tokens.
            from_file: An input file to load the text.
            corpus_type: The desired type to tokenize the text. Should be `char` or `word`.
            min_frequency: Minimum frequency of individual tokens.

        """

        super().__init__(min_frequency=min_frequency)

        if not tokens:
            text = Path(from_file).read_text(encoding="utf-8")
            self.tokens = p.tokenize(text, corpus_type)
        else:
            self.tokens = tokens

        self._check_token_frequency()
        self._build()
