"""Audio-related corpus.
"""

from collections import Counter

import nalp.utils.constants as c
import nalp.utils.loader as l
import nalp.utils.logging as log
from nalp.core import Corpus

logger = log.get_logger(__name__)


class AudioCorpus(Corpus):
    """An AudioCorpus class is used to defined the first step of the workflow.

    It serves to load the raw audio, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(self, from_file, min_frequency=1):
        """Initialization method.

        Args:
            from_file (str): An input file to load the audio.
            min_frequency (int): Minimum frequency of individual tokens.

        """

        logger.info('Overriding class: Corpus -> AudioCorpus.')

        # Overrides its parent class with any custom arguments if needed
        super(AudioCorpus, self).__init__()

        # Loads the audio from file
        audio = l.load_audio(from_file)

        # Declaring an empty list to hold audio notes
        self.tokens = []

        # Gathering notes
        for step in audio:
            # Checking for real note
            if not step.is_meta and step.channel == 0 and step.type == 'note_on':
                # Gathering note
                note = step.bytes()

                # Saving to list
                self.tokens.append(note[1])

        # Cuts the tokens based on a minimum frequency
        self._cut_tokens(min_frequency)

        # Builds the vocabulary based on the tokens
        self._build()

        # Debugging some important information
        logger.debug('Tokens: %d | Type: audio | Minimum Frequency: %d | Vocabulary Size: %d.',
                     len(self.tokens), min_frequency, len(self.vocab))
        logger.info('AudioCorpus created.')

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

    def _cut_tokens(self, min_frequency):
        """Cuts tokens that do not meet a minimum frequency value.

        Args:
            min_frequency (int): Minimum frequency of individual tokens.

        """

        # Calculates the frequency of tokens
        tokens_frequency = Counter(self.tokens)

        # Iterates over every possible sentence
        # Using index is a caveat due to lists immutable property
        for i, _ in enumerate(self.tokens):
            # If frequency of token is smaller than minimum frequency
            if tokens_frequency[self.tokens[i]] < min_frequency:
                # Replaces with an unknown token
                self.tokens[i] = c.UNK

    def _build(self):
        """Builds the vocabulary based on the tokens.

        """

        # Creates the vocabulary
        self.vocab = sorted(set(self.tokens))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}

        # Creates a property mapping indexes to vocabulary
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
