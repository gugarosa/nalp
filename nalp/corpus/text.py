import nalp.utils.loader as l
import nalp.utils.logging as log
import nalp.utils.preprocess as p
from nalp.core.corpus import Corpus

logger = log.get_logger(__name__)


class TextCorpus(Corpus):
    """A TextCorpus class is used to defined the first step of the workflow.

    It serves to load the raw text, pre-process it and create their tokens and
    vocabulary.

    """

    def __init__(self, tokens=None, from_file=None, type='char'):
        """Initialization method.

        Args:
            tokens (list): A list of tokens.
            from_file (str): An input file to load the text.
            type (str): The desired type to tokenize the text. Should be `char` or `word`.

        """

        logger.info('Overriding class: Corpus -> TextCorpus.')

        # Overrides its parent class with any custom arguments if needed
        super(TextCorpus, self).__init__()

        # Checks if there are not pre-loaded tokens
        if not tokens:
            # Loads the text from file
            text = l.load_txt(from_file)

            # Creates a pipeline based on desired type
            pipe = self._create_pipeline(type)

            # Retrieve the tokens
            self.tokens = pipe(text)

        # If there are tokens
        else:
            # Gathers them to the property
            self.tokens = tokens

        # Builds the vocabulary based on the tokens
        self._build(self.tokens)

        logger.info('TextCorpus created.')

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

    def _create_pipeline(self, type):
        """Creates a pipeline based on the input type.

        Args:
            type (str): A type to create the pipeline. Should be `char` or `word`.

        Returns:
            The created pipeline.

        """

        # Checks if type is possible
        if type not in ['char', 'word']:
            # If not, creates an error
            e = f'Type argument should be `char` or `word`.'

            # Logs the error
            logger.error(e)

            raise RuntimeError(e)

        # If the type is char
        if type == 'char':
            return p.pipeline(p.tokenize_to_char)

        # If the type is word
        elif type == 'word':
            return p.pipeline(p.tokenize_to_word)

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
