"""Text-related corpus.
"""

import nalp.utils.loader as l
import nalp.utils.logging as log
from nalp.core import Corpus

logger = log.get_logger(__name__)


class TextCorpus(Corpus):
    """A TextCorpus class is used to defined the first step of the workflow.

    It serves to load the raw text, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(self, tokens=None, from_file=None, corpus_type='char', min_frequency=1):
        """Initialization method.

        Args:
            tokens (list): A list of tokens.
            from_file (str): An input file to load the text.
            corpus_type (str): The desired type to tokenize the text. Should be `char` or `word`.
            min_frequency (int): Minimum frequency of individual tokens.

        """

        logger.info('Overriding class: Corpus -> TextCorpus.')

        super(TextCorpus, self).__init__(min_frequency=min_frequency)

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the text from file
            text = l.load_txt(from_file)

            # Creates a tokenizer based on desired type
            pipe = self._create_tokenizer(corpus_type)

            # Retrieve the tokens
            self.tokens = pipe(text)

        else:
            # Gathers them to the property
            self.tokens = tokens

        # Cuts the tokens based on a minimum frequency
        self._check_token_frequency()

        # Builds the vocabulary based on the tokens
        self._build()

        logger.debug('Tokens: %d | Minimum frequency: %d | Vocabulary size: %d.',
                     len(self.tokens), self.min_frequency, len(self.vocab))
        logger.info('TextCorpus created.')
