import nalp.utils.loader as l
import nalp.utils.logging as log
import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus

logger = log.get_logger(__name__)


class AudioCorpus(Corpus):
    """An AudioCorpus class is used to defined the first step of the workflow.

    It serves to load the raw audio, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(self, from_file=None):
        """Initialization method.

        Args:
            from_file (str): An input file to load the audio.

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

        # Builds the vocabulary based on the tokens
        self._build(self.tokens)

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
        """int: The size of the vocabulary

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

    def _build(self, tokens):
        """Builds the vocabulary based on the tokens.

        Args:
            tokens (list): A list of tokens.

        """

        # Creates the vocabulary
        self.vocab = sorted(set(tokens))

        # Also, gathers the vocabulary size
        self.vocab_size = len(self.vocab)

        # Creates a property mapping vocabulary to indexes
        self.vocab_index = {t: i for i, t in enumerate(self.vocab)}

        # Creates a property mapping indexes to vocabulary
        self.index_vocab = {i: t for i, t in enumerate(self.vocab)}
