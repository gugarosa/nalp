"""Sentence-related corpus.
"""

from collections import Counter
from itertools import chain

import nalp.utils.constants as c
import nalp.utils.loader as l
import nalp.utils.logging as log
from nalp.core import Corpus

logger = log.get_logger(__name__)


class SentenceCorpus(Corpus):
    """A SentenceCorpus class is used to defined the first step of the workflow.

    It serves to load the raw sentences, pre-process them and create their tokens and
    vocabulary.

    """

    def __init__(self, tokens=None, from_file=None, corpus_type='char', min_frequency=1,
                 max_pad_length=None, sos_eos_tokens=True):
        """Initialization method.

        Args:
            tokens (list): A list of tokens.
            from_file (str): An input file to load the sentences.
            corpus_type (str): The desired type to tokenize the sentences. Should be `char` or `word`.
            min_frequency (int): Minimum frequency of individual tokens.
            max_pad_length (int): Maximum length to pad the tokens.
            sos_eos_tokens (bool): Whether start-of-sentence and end-of-sentence tokens should be used.

        """

        logger.info('Overriding class: Corpus -> SentenceCorpus.')

        super(SentenceCorpus, self).__init__(min_frequency=min_frequency)

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the sentences from file
            sentences = l.load_txt(from_file).splitlines()

            # Creates a tokenizer based on desired type
            pipe = self._create_tokenizer(corpus_type)

            # Retrieve the tokens
            self.tokens = [pipe(sentence) for sentence in sentences]

        else:
            # Gathers them to the property
            self.tokens = tokens

        # Cuts the tokens based on a minimum frequency
        self._check_token_frequency()

        # Pads the tokens before building the vocabulary
        self._pad_token(max_pad_length, sos_eos_tokens)

        # Builds the vocabulary based on the tokens
        self._build()

        logger.debug('Sentences: %d | Minimum frequency: %d | Maximum pad length: %s | '
                     'Use <SOS> and <EOS>: %s | Vocabulary size: %d.',
                     len(self.tokens), self.min_frequency, max_pad_length,
                     sos_eos_tokens, len(self.vocab))
        logger.info('SentenceCorpus created.')

    def _check_token_frequency(self):
        """Cuts tokens that do not meet a minimum frequency value.

        """

        # Calculates the frequency of tokens
        tokens_frequency = Counter(chain.from_iterable(self.tokens))

        # Iterates over every possible sentence
        # Using index is a caveat due to lists immutable property
        for i, _ in enumerate(self.tokens):
            # Iterates over every token in the sentence
            for j, _ in enumerate(self.tokens[i]):
                # If frequency of token is smaller than minimum frequency
                if tokens_frequency[self.tokens[i][j]] < self.min_frequency:
                    # Replaces with an unknown token
                    self.tokens[i][j] = c.UNK

    def _pad_token(self, max_pad_length, sos_eos_tokens):
        """Pads the tokens into a fixed length.

        Args:
            max_pad_length (int): Maximum length to pad the tokens.
            sos_eos_tokens (bool): Whether start-of-sentence and end-of-sentence tokens should be used.

        """

        if not max_pad_length:
            max_pad_length = len(max(self.tokens, key=lambda t: len(t)))

        # Iterates over every possible sentence
        # Using index is a caveat due to lists immutable property
        for i, _ in enumerate(self.tokens):
            # Gathers the difference between length of current token and maximum length
            length_diff = max_pad_length - len(self.tokens[i])

            if length_diff > 0:
                # Pads the input based on the remaining tokens
                self.tokens[i] += [c.PAD] * length_diff

            else:
                # Gathers the maximum length allowed
                self.tokens[i] = self.tokens[i][:max_pad_length]

            if sos_eos_tokens:
                self.tokens[i].insert(0, c.SOS)
                self.tokens[i].append(c.EOS)

    def _build(self):
        """Builds the vocabulary based on the tokens.

        """

        # Creates the vocabulary
        self.vocab = sorted(set(chain.from_iterable(self.tokens)).union({c.UNK}))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes and vice-versa
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
