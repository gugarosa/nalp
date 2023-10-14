"""Text-related corpus.
"""

from typing import List, Optional

from nalp.core import Corpus
from nalp.utils import loader, logging

logger = logging.get_logger(__name__)


class TextCorpus(Corpus):
    """A TextCorpus class is used to defined the first step of the workflow.

    It serves to load the raw text, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        from_file: Optional[str] = None,
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

        logger.info("Overriding class: Corpus -> TextCorpus.")

        super(TextCorpus, self).__init__(min_frequency=min_frequency)

        if not tokens:
            text = loader.load_txt(from_file)

            pipe = self._create_tokenizer(corpus_type)
            self.tokens = pipe(text)
        else:
            self.tokens = tokens

        self._check_token_frequency()
        self._build()

        logger.debug(
            "Tokens: %d | Minimum frequency: %d | Vocabulary size: %d.",
            len(self.tokens),
            self.min_frequency,
            len(self.vocab),
        )
        logger.info("TextCorpus created.")
