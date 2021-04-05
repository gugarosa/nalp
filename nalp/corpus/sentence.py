"""Sentence-related corpus.
"""

from itertools import chain

import nalp.utils.loader as l
import nalp.utils.logging as log
import nalp.utils.preprocess as p
from nalp.core import Corpus

logger = log.get_logger(__name__)


class SentenceCorpus(Corpus):
    """A SentenceCorpus class is used to defined the first step of the workflow.

    It serves to load the raw sentences, pre-process them and create their tokens and
    vocabulary.

    """

    def __init__(self, tokens=None, from_file=None, corpus_type='char', max_pad_length=None, sos_eos_tokens=True):
        """Initialization method.

        Args:
            tokens (list): A list of tokens.
            from_file (str): An input file to load the sentences.
            corpus_type (str): The desired type to tokenize the sentences. Should be `char` or `word`.
            max_pad_length (int): Maximum length to pad the tokens.
            sos_eos_tokens (bool): Whether start-of-sentence and end-of-sentence tokens should be used.

        """

        logger.info('Overriding class: Corpus -> SentenceCorpus.')

        # Overrides its parent class with any custom arguments if needed
        super(SentenceCorpus, self).__init__()

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the sentences from file
            sentences = l.load_txt(from_file).splitlines()

            # Creates a tokenizer based on desired type
            pipe = self._create_tokenizer(corpus_type)

            # Retrieve the tokens
            self.tokens = [pipe(sentence) for sentence in sentences]

        # If there are tokens
        else:
            # Gathers them to the property
            self.tokens = tokens

        # Pads the tokens before building the vocabulary
        self._pad_tokens(max_pad_length, sos_eos_tokens)

        # Builds the vocabulary based on the tokens
        self._build()

        # Debugging some important information
        logger.debug('Sentences: %d | Vocabulary Size: %d | Type: %s.',
                     len(self.tokens), len(self.vocab), corpus_type)
        logger.info('SentenceCorpus created.')

    @property
    def vocab(self):
        """list: The vocabulary itself.

        """

        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab

    @property
    def vocab_size(self):
        """int: The size of the vocabulary.

        """

        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self._vocab_size = vocab_size

    @property
    def vocab_index(self):
        """dict: A dictionary mapping vocabulary to indexes.

        """

        return self._vocab_index

    @vocab_index.setter
    def vocab_index(self, vocab_index):
        self._vocab_index = vocab_index

    @property
    def index_vocab(self):
        """dict: A dictionary mapping indexes to vocabulary.

        """

        return self._index_vocab

    @index_vocab.setter
    def index_vocab(self, index_vocab):
        self._index_vocab = index_vocab

    def _create_tokenizer(self, corpus_type):
        """Creates a tokenizer based on the input type.

        Args:
            corpus_type (str): A type to create the tokenizer. Should be `char` or `word`.

        Returns:
            The created tokenizer.

        """

        # Checks if type is possible
        if corpus_type not in ['char', 'word']:
            # If not, creates an error
            e = 'Type argument should be `char` or `word`.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # If the type is char
        if corpus_type == 'char':
            return p.pipeline(p.tokenize_to_char)

        # If the type is word
        if corpus_type == 'word':
            return p.pipeline(p.tokenize_to_word)

    def _pad_tokens(self, max_pad_length, sos_eos_tokens):
        """Pads the tokens into a fixed length.

        Args:
            max_pad_length (int): Maximum length to pad the tokens.
            sos_eos_tokens (bool): Whether start-of-sentence and end-of-sentence tokens should be used.

        """

        # Checks if there is a supplied maximum pad length
        if not max_pad_length:
            # Gathers the maximum length to pad the tokens
            max_pad_length = len(max(self.tokens, key=lambda t: len(t)))

        # Iterates over every possible token
        # Using index is a caveat due to lists immutable property
        for i, _ in enumerate(self.tokens):
            # Gathers the difference between length of current token and maximum length
            length_diff = max_pad_length - len(self.tokens[i])

            # If length difference is bigger than zero
            if length_diff > 0:
                # Pads the input based on the remaining tokens
                self.tokens[i] += ['<PAD>'] * length_diff

            # If length difference is smaller or equal to zero
            else:
                # Gathers the maximum length allowed
                self.tokens[i] = self.tokens[i][:max_pad_length]

            # Checks if additional tokens should be added
            if sos_eos_tokens:
                # Adds start-of-sentence and end-of-sentence tokens
                self.tokens[i].insert(0, '<SOS>')
                self.tokens[i].append('<EOS>')

    def _build(self):
        """Builds the vocabulary based on the tokens.

        """

        # Creates the vocabulary
        self.vocab = sorted(set(chain.from_iterable(self.tokens)).union({'<UNK>'}))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}

        # Creates a property mapping indexes to vocabulary
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
